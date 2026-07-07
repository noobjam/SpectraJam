from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from heapq import merge
from itertools import groupby
from typing import Iterable

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry


@dataclass(frozen=True, slots=True)
class PixelCell:
    epsg: int
    x_index: int
    y_index: int
    resolution_m: int = 10

    @property
    def pixel_id(self) -> str:
        return f"utm-{self.epsg}-{self.resolution_m}m-{self.x_index}-{self.y_index}"

    @property
    def x(self) -> float:
        return (self.x_index + 0.5) * self.resolution_m

    @property
    def y(self) -> float:
        return (self.y_index + 0.5) * self.resolution_m


@dataclass(frozen=True, slots=True)
class WorkTileKey:
    epsg: int
    x_index: int
    y_index: int

    @property
    def key(self) -> str:
        return f"epsg{self.epsg}-x{self.x_index}-y{self.y_index}"


@dataclass(frozen=True, slots=True)
class RasterChunk:
    epsg: int
    x_index: int
    y_index: int
    cells: int
    resolution_m: int

    @property
    def left(self) -> float:
        return self.x_index * self.cells * self.resolution_m

    @property
    def bottom(self) -> float:
        return self.y_index * self.cells * self.resolution_m

    @property
    def right(self) -> float:
        return self.left + self.cells * self.resolution_m

    @property
    def top(self) -> float:
        return self.bottom + self.cells * self.resolution_m

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.left, self.bottom, self.right, self.top

    @property
    def key(self) -> str:
        return f"epsg{self.epsg}-cx{self.x_index}-cy{self.y_index}"

    def local_indices(self, pixel: PixelCell) -> tuple[int, int]:
        if pixel.epsg != self.epsg or pixel.resolution_m != self.resolution_m:
            raise ValueError("pixel and raster chunk use different grids")
        column = pixel.x_index - self.x_index * self.cells
        bottom_up_row = pixel.y_index - self.y_index * self.cells
        row = self.cells - 1 - bottom_up_row
        if not (0 <= row < self.cells and 0 <= column < self.cells):
            raise ValueError(f"pixel {pixel.pixel_id} lies outside chunk {self.key}")
        return row, column


@dataclass(frozen=True, slots=True)
class RasterWindow:
    """Smallest globally snapped rectangle containing one task's pixels."""

    epsg: int
    left_x_index: int
    bottom_y_index: int
    width: int
    height: int
    resolution_m: int

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("raster window dimensions must be positive")

    @classmethod
    def from_pixels(cls, pixels: Iterable[PixelCell]) -> RasterWindow:
        values = tuple(pixels)
        if not values:
            raise ValueError("at least one pixel is required for a raster window")
        epsg = values[0].epsg
        resolution = values[0].resolution_m
        if any(pixel.epsg != epsg or pixel.resolution_m != resolution for pixel in values):
            raise ValueError("all raster-window pixels must use one grid")
        x_values = [pixel.x_index for pixel in values]
        y_values = [pixel.y_index for pixel in values]
        left = min(x_values)
        bottom = min(y_values)
        return cls(
            epsg=epsg,
            left_x_index=left,
            bottom_y_index=bottom,
            width=max(x_values) - left + 1,
            height=max(y_values) - bottom + 1,
            resolution_m=resolution,
        )

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left = self.left_x_index * self.resolution_m
        bottom = self.bottom_y_index * self.resolution_m
        return (
            left,
            bottom,
            left + self.width * self.resolution_m,
            bottom + self.height * self.resolution_m,
        )

    def local_indices(self, pixel: PixelCell) -> tuple[int, int]:
        if pixel.epsg != self.epsg or pixel.resolution_m != self.resolution_m:
            raise ValueError("pixel and raster window use different grids")
        column = pixel.x_index - self.left_x_index
        bottom_up_row = pixel.y_index - self.bottom_y_index
        row = self.height - 1 - bottom_up_row
        if not (0 <= row < self.height and 0 <= column < self.width):
            raise ValueError(f"pixel {pixel.pixel_id} lies outside raster window")
        return row, column


def utm_epsg(longitude: float, latitude: float) -> int:
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError(f"invalid longitude: {longitude}")
    if not math.isfinite(latitude) or not -80 <= latitude <= 84:
        raise ValueError(f"latitude is outside UTM coverage: {latitude}")
    normalized = min(longitude, math.nextafter(180.0, -math.inf))
    zone = int(math.floor((normalized + 180.0) / 6.0)) + 1
    return (32600 if latitude >= 0 else 32700) + zone


def _polygon_parts(geometry: BaseGeometry) -> list[BaseGeometry]:
    if geometry.geom_type == "Polygon":
        return [geometry]
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        parts: list[BaseGeometry] = []
        for child in shapely.get_parts(geometry):
            parts.extend(_polygon_parts(child))
        return parts
    return []


def parse_field_geometry(value: object) -> tuple[BaseGeometry, str]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("field WKT is empty")
    try:
        geometry = shapely.from_wkt(value)
    except Exception as error:
        raise ValueError("field WKT cannot be parsed") from error
    if geometry is None or geometry.is_empty:
        raise ValueError("field geometry is empty")
    status = "valid"
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
        status = "repaired"
    parts = _polygon_parts(geometry)
    if not parts:
        raise ValueError(f"field geometry must be Polygon/MultiPolygon, got {geometry.geom_type}")
    polygon = shapely.union_all(parts)
    if polygon.is_empty:
        raise ValueError("field geometry has no polygonal area")
    return polygon, status


def canonical_geometry_sha256(geometry: BaseGeometry) -> str:
    normalized = shapely.normalize(geometry)
    encoded = shapely.to_wkb(normalized, hex=False, output_dimension=2, byte_order=1)
    return hashlib.sha256(encoded).hexdigest()


def project_geometry(geometry: BaseGeometry, destination_epsg: int) -> BaseGeometry:
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
    transformer = Transformer.from_crs(4326, destination_epsg, always_xy=True)
    return shapely.transform(geometry, transformer.transform, interleaved=False)


def pixel_cells_for_geometry(
    projected_geometry: BaseGeometry,
    epsg: int,
    resolution_m: int = 10,
    scan_rows: int = 512,
) -> tuple[PixelCell, ...]:
    """Return globally snapped cells whose centers are covered by a polygon."""
    min_x, min_y, max_x, max_y = projected_geometry.bounds
    x_start = math.floor(min_x / resolution_m)
    x_stop = math.ceil(max_x / resolution_m)
    y_start = math.floor(min_y / resolution_m)
    y_stop = math.ceil(max_y / resolution_m)
    if x_stop <= x_start or y_stop <= y_start:
        return ()

    x_indices = np.arange(x_start, x_stop, dtype=np.int64)
    x_centers = (x_indices.astype(np.float64) + 0.5) * resolution_m
    selected: list[PixelCell] = []
    for block_start in range(y_start, y_stop, scan_rows):
        block_stop = min(block_start + scan_rows, y_stop)
        y_indices = np.arange(block_start, block_stop, dtype=np.int64)
        y_centers = (y_indices.astype(np.float64) + 0.5) * resolution_m
        xx, yy = np.meshgrid(x_centers, y_centers)
        covered = shapely.intersects_xy(projected_geometry, xx, yy)
        rows, columns = np.nonzero(covered)
        selected.extend(
            PixelCell(epsg, int(x_indices[column]), int(y_indices[row]), resolution_m)
            for row, column in zip(rows, columns, strict=True)
        )
    return tuple(sorted(selected, key=lambda cell: (cell.y_index, cell.x_index)))


def positive_area_pixel_count(
    projected_geometry: BaseGeometry,
    resolution_m: int = 10,
    block_pixels: int = 256,
) -> int:
    """Count grid cells whose footprint overlaps the geometry with positive area."""
    if resolution_m <= 0:
        raise ValueError("resolution_m must be positive")
    if block_pixels < 1:
        raise ValueError("block_pixels must be positive")
    parts = _polygon_parts(projected_geometry)
    if not parts:
        return 0

    polygonal_geometry = (
        projected_geometry
        if projected_geometry.geom_type in {"Polygon", "MultiPolygon"}
        else shapely.union_all(parts)
    )

    def candidate_blocks(part: BaseGeometry):
        min_x, min_y, max_x, max_y = part.bounds
        x_start = math.floor(min_x / resolution_m)
        x_stop = math.ceil(max_x / resolution_m)
        y_start = math.floor(min_y / resolution_m)
        y_stop = math.ceil(max_y / resolution_m)
        if x_stop <= x_start or y_stop <= y_start:
            return
        for y_block in range(
            y_start // block_pixels,
            (y_stop - 1) // block_pixels + 1,
        ):
            block_bottom = y_block * block_pixels
            local_y_start = max(y_start, block_bottom) - block_bottom
            local_y_stop = min(y_stop, block_bottom + block_pixels) - block_bottom
            for x_block in range(
                x_start // block_pixels,
                (x_stop - 1) // block_pixels + 1,
            ):
                block_left = x_block * block_pixels
                local_x_start = max(x_start, block_left) - block_left
                local_x_stop = min(x_stop, block_left + block_pixels) - block_left
                yield (
                    y_block,
                    x_block,
                    local_y_start,
                    local_y_stop,
                    local_x_start,
                    local_x_stop,
                )

    shapely.prepare(polygonal_geometry)
    leaf_cell_limit = min(block_pixels * block_pixels, 1024)

    def count_rectangle(x_start: int, x_stop: int, y_start: int, y_stop: int) -> int:
        left = x_start * resolution_m
        right = x_stop * resolution_m
        bottom = y_start * resolution_m
        top = y_stop * resolution_m
        footprint = shapely.Polygon(
            ((left, bottom), (right, bottom), (right, top), (left, top))
        )
        if shapely.covers(polygonal_geometry, footprint):
            return (x_stop - x_start) * (y_stop - y_start)
        if not shapely.intersects(polygonal_geometry, footprint):
            return 0

        width = x_stop - x_start
        height = y_stop - y_start
        if width * height > leaf_cell_limit:
            if width >= height and width > 1:
                middle = x_start + width // 2
                return count_rectangle(
                    x_start, middle, y_start, y_stop
                ) + count_rectangle(middle, x_stop, y_start, y_stop)
            middle = y_start + height // 2
            return count_rectangle(x_start, x_stop, y_start, middle) + count_rectangle(
                x_start, x_stop, middle, y_stop
            )

        x_indices = np.arange(x_start, x_stop, dtype=np.float64)
        y_indices = np.arange(y_start, y_stop, dtype=np.float64)
        lefts = x_indices * resolution_m
        bottoms = y_indices * resolution_m
        cells = shapely.box(
            lefts[None, :],
            bottoms[:, None],
            lefts[None, :] + resolution_m,
            bottoms[:, None] + resolution_m,
        )
        intersects = shapely.intersects(polygonal_geometry, cells)
        intersecting_cells = cells[intersects]
        if not intersecting_cells.size:
            return 0
        # For polygonal inputs, non-touching intersections have overlapping
        # interiors and therefore strictly positive intersection area.
        return int(
            intersecting_cells.size
            - np.count_nonzero(shapely.touches(polygonal_geometry, intersecting_cells))
        )

    count = 0
    streams = (candidate_blocks(part) for part in parts)
    blocks = groupby(merge(*streams), key=lambda candidate: candidate[:2])
    for (y_block, x_block), candidates in blocks:
        local_y_start = block_pixels
        local_y_stop = 0
        local_x_start = block_pixels
        local_x_stop = 0
        for _, _, y_start, y_stop, x_start, x_stop in candidates:
            local_y_start = min(local_y_start, y_start)
            local_y_stop = max(local_y_stop, y_stop)
            local_x_start = min(local_x_start, x_start)
            local_x_stop = max(local_x_stop, x_stop)
        block_bottom = y_block * block_pixels
        block_left = x_block * block_pixels
        count += count_rectangle(
            block_left + local_x_start,
            block_left + local_x_stop,
            block_bottom + local_y_start,
            block_bottom + local_y_stop,
        )
    return count


def work_tile_for_pixel(pixel: PixelCell, work_tile_m: int) -> WorkTileKey:
    cells = work_tile_m // pixel.resolution_m
    return WorkTileKey(pixel.epsg, pixel.x_index // cells, pixel.y_index // cells)


def raster_chunk_for_pixel(pixel: PixelCell, chunk_cells: int) -> RasterChunk:
    return RasterChunk(
        epsg=pixel.epsg,
        x_index=pixel.x_index // chunk_cells,
        y_index=pixel.y_index // chunk_cells,
        cells=chunk_cells,
        resolution_m=pixel.resolution_m,
    )


def work_tile_bounds(
    key: WorkTileKey, work_tile_m: int
) -> tuple[float, float, float, float]:
    left = key.x_index * work_tile_m
    bottom = key.y_index * work_tile_m
    return left, bottom, left + work_tile_m, bottom + work_tile_m


def expand_projected_bounds(
    bounds: tuple[float, float, float, float], halo_m: float
) -> tuple[float, float, float, float]:
    """Expand projected query bounds without changing a field geometry."""
    if not math.isfinite(halo_m) or halo_m < 0:
        raise ValueError("halo_m must be finite and non-negative")
    left, bottom, right, top = bounds
    if not all(math.isfinite(value) for value in bounds) or right < left or top < bottom:
        raise ValueError("projected bounds are invalid")
    return left - halo_m, bottom - halo_m, right + halo_m, top + halo_m


def projected_bounds_to_wgs84(
    bounds: tuple[float, float, float, float], epsg: int
) -> tuple[float, float, float, float]:
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
    transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
    transformed = transformer.transform_bounds(*bounds, densify_pts=21)
    return tuple(float(value) for value in transformed)


def pixel_centers_wgs84(cells: Iterable[PixelCell]) -> dict[str, tuple[float, float]]:
    grouped: dict[int, list[PixelCell]] = {}
    for cell in cells:
        grouped.setdefault(cell.epsg, []).append(cell)
    result: dict[str, tuple[float, float]] = {}
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
    for epsg, group in grouped.items():
        transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
        longitude, latitude = transformer.transform(
            [cell.x for cell in group],
            [cell.y for cell in group],
        )
        result.update(
            {
                cell.pixel_id: (float(lon), float(lat))
                for cell, lon, lat in zip(group, longitude, latitude, strict=True)
            }
        )
    return result
