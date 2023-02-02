use std::cmp::max;
use std::cmp::min;
use std::cmp::Ordering;

use ndarray::prelude::*;
use numpy::PyReadonlyArrayDyn;
use pyo3::prelude::*;

// type  Slice = (usize, (usize, (usize, usize))); // (z, (y, (x_min, x_max)))

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Slice {
    z: usize,
    y: usize,
    s: usize,
    e: usize,
}

impl Slice {
    fn values(&self) -> [usize; 4] {
        [self.z, self.y, self.s, self.e]
    }

    fn _overlaps(&self, other: &Slice, connectivity: u8) -> bool {
        let [lz, ly, ls, le] = self.values();
        let [rz, ry, rs, re] = other.values();
    
        let dz = if lz > rz { lz - rz } else { rz - lz };
        let dy = if ly > ry { ly - ry } else { ry - ly };
    
        if (dz > 1) || (dy > 1) || (ls > re) || (rs > le) {
            return false;
        }
    
        if dz == 0 && dy == 0 {
            (ls == re) || (rs == le)
        } else if dy == 1 && dz == 1 {
            if connectivity == 3 {
                (ls <= re) || (rs <= le)
            } else {
                (ls < re) || (rs < le)
            }
        } else if connectivity == 1 {
            (ls < re) || (rs < le)
        } else {
            (ls <= re) || (rs <= le)
        }
    }
    
    #[inline(never)]
    fn do_slices_overlap(&self, other: &Self, connectivity: u8) -> bool {
        if self > other {
            other._overlaps(self, connectivity)
        } else {
            self._overlaps(other, connectivity)
        }
    }

}

/// A `Polygon` represents a single connected object to be labelled.
#[derive(Debug, PartialEq, Eq)]
pub struct Polygon {
    /// Connectivity determines how we find neighbors for pixels.
    connectivity: u8,
    /// A collection of slices that represents the exact bounds of the polygon.
    slices: Vec<Slice>,
    /// minimum x-value of the bounding-box around the polygon.
    x_min: usize,
    /// maximum x-value of the bounding-box around the polygon.
    x_max: usize,
    /// minimum y-value of the bounding-box around the polygon.
    y_min: usize,
    /// maximum y-value of the bounding-box around the polygon.
    y_max: usize,
    /// minimum z-value of the bounding-box around the polygon.
    z_min: usize,
    /// maximum z-value of the bounding-box around the polygon.
    z_max: usize,
}

impl Polygon {
    /// Creates a new `Polygon` from the given slices.
    ///
    /// # Arguments
    ///
    /// * `connectivity` - A `u8` to represent the connectivity used for creating a `Polygon`.
    /// * `slices` - A Vec of Slices that represent the exact boundaries of the `Polygon`.
    ///              This must be non-empty. We assume that the caller has verified that the slices are connected.
    ///              Each slice is represented as a nested tuple `(z, (y, (x_min, x_max)))`
    fn new(connectivity: u8, slices: Vec<Slice>) -> Self {
        assert!(
            !slices.is_empty(),
            "Cannot create Polygon without any slices."
        );

        let z_min = slices.iter().map(|s| s.z).min().unwrap();
        let z_max = slices.iter().map(|s| s.z).max().unwrap() + 1;

        let y_min = slices.iter().map(|s| s.y).min().unwrap();
        let y_max = slices.iter().map(|s| s.y).max().unwrap() + 1;

        let x_min = slices.iter().map(|s| s.s).min().unwrap();
        let x_max = slices.iter().map(|s| s.e).max().unwrap() + 1;

        Polygon {
            connectivity,
            slices,
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        }
    }

    /// Returns whether this `Polygon` has any slices.
    pub fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    /// Returns the number of slices in this `Polygon`.
    pub fn len(&self) -> usize {
        self.slices.len()
    }

    /// Returns whether this `Polygon's` bounding-box intersects with that of another `Polygon`.
    ///
    /// This is useful as an early filter for the `boundary_connects` method.
    ///
    /// # Arguments
    ///
    /// * `other` - Another `Polygon`.
    ///
    /// # Returns
    ///
    /// * Whether the bounding-boxes of the two `Polygons` overlap.
    pub fn bbox_overlap(&self, other: &Self) -> bool {
        self.x_min <= other.x_max
            && self.y_min <= other.y_max
            && self.z_min <= other.z_max
            && self.x_max >= other.x_min
            && self.y_max >= other.y_min
            && self.z_max >= other.z_min
    }

    /// Returns whether this `Polygon` connects with another `Polygon`.
    ///
    /// # Arguments
    ///
    /// * `other` - Another `Polygon`.
    ///1
    /// # Returns
    ///
    /// * Whether the two `Polygons` intersect.
    #[inline(never)]
    pub fn boundary_connects(&self, other: &Self) -> bool {
        // TODO: rayon
        self.bbox_overlap(other)
            && self.slices.iter().any(|left| {
                other
                    .slices
                    .iter()
                    .any(|right| left.do_slices_overlap(right, self.connectivity))
            })
    }

    /// Absorbs the other `Polygon` into itself.
    ///
    /// This leaves the other `Polygon` empty. The caller is responsible for properly handling the other `Polygon`.
    #[inline(never)]
    pub fn absorb(&mut self, others: Vec<Self>) {
        assert!(!others.is_empty());
        self.x_min = min(self.x_min, others.iter().map(|o| o.x_min).min().unwrap());
        self.y_min = min(self.y_min, others.iter().map(|o| o.y_min).min().unwrap());
        self.z_min = min(self.z_min, others.iter().map(|o| o.z_min).min().unwrap());
        self.x_max = max(self.x_max, others.iter().map(|o| o.x_max).max().unwrap());
        self.y_max = max(self.y_max, others.iter().map(|o| o.y_max).max().unwrap());
        self.z_max = max(self.z_max, others.iter().map(|o| o.z_max).max().unwrap());

        self.slices
            .extend(others.into_iter().flat_map(|o| o.slices.into_iter()))
    }
}

/// Given a `Vec` of `Polygons` and a connectivity, partitions the `Polygons`
/// into groups that are connected, merges each group into a single polygon, and
/// returns the merged polygons.
///
/// # Arguments
///
/// `polygons` - A `Vec` of `Polygons` to process. This Vec will be consumed by
/// the function.
/// `connectivity` - A `u8` to represent the connectivity used for determining
/// neighbors.
///
/// # Returns
///
/// A `Vec` of the merged `Polygons`. These `Polygons` do not connect with each
/// other.
#[inline(never)]
fn bft_partition(polygons: Vec<Polygon>) -> Vec<Polygon> {
    // TODO: This takes ~91% of the time
    polygons.into_iter().fold(Vec::new(), |mut merged, mut t| {
        let (to_be_merged, mut merged): (Vec<_>, Vec<_>) =
            merged.drain(..).partition(|c| t.boundary_connects(c));
        if !to_be_merged.is_empty() {
            t.absorb(to_be_merged);
        }
        merged.push(t);
        merged
    })
    // .reduce(Vec::new, |mut a, b| {
    //     // a.append(&mut b);
    //     a.extend(b.into_iter());
    //     a
    // })
}

impl Ord for Polygon {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Polygon {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match self.z_min.cmp(&other.z_min) {
            Ordering::Less => Some(Ordering::Less),
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Equal => match self.y_min.cmp(&other.y_min) {
                Ordering::Less => Some(Ordering::Less),
                Ordering::Greater => Some(Ordering::Greater),
                Ordering::Equal => match self.x_min.cmp(&other.x_min) {
                    Ordering::Less => Some(Ordering::Less),
                    Ordering::Greater => Some(Ordering::Greater),
                    Ordering::Equal => match self.z_max.cmp(&other.z_max) {
                        Ordering::Less => Some(Ordering::Less),
                        Ordering::Greater => Some(Ordering::Greater),
                        Ordering::Equal => match self.y_max.cmp(&other.y_max) {
                            Ordering::Less => Some(Ordering::Less),
                            Ordering::Greater => Some(Ordering::Greater),
                            Ordering::Equal => Some(self.x_max.cmp(&other.x_max)),
                        },
                    },
                },
            },
        }
    }
}

/// A `PolygonSet` handles and maintains `Polygons` in an image. This provides
/// utilities for adding tiles from an image,reconciling labels within and
/// across tiles, and extracting tiles with labelled objects.
#[pyclass]
#[derive(Clone, Debug)]
pub struct PolygonSet {
    /// A `u8` to represent the connectivity used for determining neighbors.
    connectivity: u8,
    polygons: std::sync::Arc<std::sync::RwLock<Vec<Polygon>>>,
}

#[pymethods]
impl PolygonSet {
    /// Creates a new `PolygonSet` with an empty `Vec` of `Polygons`.
    #[new]
    pub fn new(connectivity: u8) -> Self {
        PolygonSet {
            connectivity,
            polygons: Default::default(),
        }
    }

    /// Returns whether the `PolygonSet` is empty. This method is a Rust-recommended complement to the `len` method.
    pub fn is_empty(&self) -> bool {
        self.polygons.read().unwrap().is_empty()
    }

    /// Returns the number of `Polygons` in this set.
    pub fn len(&self) -> usize {
        self.polygons.read().unwrap().len()
    }

    pub fn add_tile(
        &mut self,
        tile: PyReadonlyArrayDyn<u8>,
        top_left_point: (usize, usize, usize),
    ) {
        self._add_tile(tile.as_array(), top_left_point)
    }

    /// Restores the invariant that no two `Polygons` in the `PolygonSet` connect with each other.
    pub fn digest(&self) {
        let mut p = bft_partition(self.polygons.write().unwrap().drain(..).collect());
        p.sort();
        *self.polygons.write().unwrap() = p;
    }
}

impl PolygonSet {
    /// Detects `Polygons` in a tile and adds them to the set.
    ///
    /// This might break the invariant that no two `Polygons` in the `PolygonSet` connect with each other.
    /// Therefore, The user must call the `digest` method after all tiles have been added.
    pub fn _add_tile(&self, tile: ArrayViewD<u8>, top_left_point: (usize, usize, usize)) {
        // TODO: Figure out how to add tiles in parallel, with the GIL being the main problem.
        // TODO: Is it possible to have Rust directly call methods from bfio?
        let (z_min, y_min, x_min) = top_left_point;

        // TODO: rayon
        let slices = tile
            .outer_iter()
            .enumerate()
            .flat_map(|(z, plane)| {
                let slices = plane
                    .outer_iter()
                    .enumerate()
                    .flat_map(|(y, row)| {
                        // TODO: Insert AVX here.
                        let runs: Vec<_> = row
                            .iter()
                            .chain([0].iter())
                            .zip([0].iter().chain(row.iter()))
                            .enumerate()
                            .filter(|(_, (&a, &b))| a != b)
                            .map(|(i, _)| i)
                            .collect();

                        if runs.is_empty() {
                            Vec::new()
                        } else {
                            let (starts, ends): (Vec<_>, Vec<_>) =
                                runs.into_iter().enumerate().partition(|&(i, _)| i % 2 == 0);
                            // let starts: Vec<_> = runs.iter().step_by(2).copied().collect();
                            // let ends: Vec<_> = runs.into_iter().skip(1).step_by(2).collect();

                            // TODO: rayon
                            starts
                                .into_iter()
                                .map(|(_, s)| s)
                                .zip(ends.into_iter().map(|(_, s)| s))
                                .map(|(start, stop)| {
                                    Polygon::new(
                                        self.connectivity,
                                        vec![Slice {
                                            z: z + z_min,
                                            y: y + y_min,
                                            s: start + x_min,
                                            e: stop + x_min,
                                        }]
                                    )
                                })
                                .collect::<Vec<_>>()
                        }
                    })
                    .collect::<Vec<_>>();
                bft_partition(slices)
            })
            .collect::<Vec<_>>();
        self.polygons
            .write()
            .unwrap()
            .append(&mut bft_partition(slices));
    }

    /// Once all tiles have been added and digested, this method can be used to extract tiles with properly labelled objects.
    pub fn _extract_tile(
        &self,
        coordinates: (usize, usize, usize, usize, usize, usize),
    ) -> ArrayD<usize> {
        let (z_min, z_max, y_min, y_max, x_min, x_max) = coordinates;

        let tile_polygon = Polygon::new(
            self.connectivity,
            vec![
                Slice { z: z_min, y: y_min, s: x_min, e: x_max },
                Slice { z: z_max, y: y_max, s: x_min, e: x_max },
            ],
        );

        // We use a Read-Write lock on each row to allow writing to rows in parallel.
        // This really only shines when we try to extract very large tiles.
        // Otherwise, it is no slower than a single-threaded implementation without the RwLocks.
        let mut tile: Array3<usize> = Array3::zeros((z_max - z_min, y_max - y_min, x_max - x_min));

        self.polygons
            .read()
            .unwrap()
            .iter()
            .enumerate()
            .for_each(|(i, polygon)| {
                if tile_polygon.bbox_overlap(polygon) {
                    polygon
                        .slices
                        .iter()
                        .map(|s| s.values())
                        .for_each(|[z, y, start, stop]| {
                            if z_min <= z && z < z_max && y_min <= y && y < y_max {
                                let start = max(x_min, start);
                                let stop = min(x_max, stop);
                                let mut section = tile.slice_mut(s![
                                    z - z_min,
                                    y - y_min,
                                    (start - x_min)..(stop - x_min)
                                ]);
                                section.fill(i + 1);
                            }
                        });
                }
            });

        tile.into_dyn()
    }
}
