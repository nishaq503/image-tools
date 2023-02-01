use std::path::PathBuf;

use ftl_rust::PolygonSet;
use memmap2::Mmap;
use ndarray::prelude::*;
use ndarray_npy::ViewNpyExt;

fn main() {
    let tile_size = 1024;

    let mmap = {
        let mut path: PathBuf = std::env::current_dir().unwrap();
        path.push("data");
        path.push("input");
        path.push("t1_x1.npy");

        let path = path.canonicalize().unwrap();
        println!("reading path {path:?}");
        let file = std::fs::File::open(path).unwrap();
        unsafe { Mmap::map(&file).unwrap() }
    };
    let data = ArrayView3::<u8>::view_npy(&mmap).unwrap();
    println!("Shape: {:?}", data.shape());

    let y_shape = data.shape()[1];
    let x_shape = data.shape()[2];

    let ys = (0..y_shape).step_by(tile_size).collect::<Vec<_>>();
    let xs = (0..x_shape).step_by(tile_size).collect::<Vec<_>>();

    let mut polygon_set = PolygonSet::new(1);
    ys.iter()
        .enumerate()
        .filter(|(iy, _)| (3..7).contains(iy))
        .for_each(|(iy, &y)| {
            let y_max = std::cmp::min(y_shape, y + tile_size);

            xs.iter()
                .enumerate()
                .filter(|(ix, _)| (3..7).contains(ix))
                .for_each(|(ix, &x)| {
                    println!("Tile index: {iy}, {ix}");
                    let x_max = std::cmp::min(x_shape, x + tile_size);

                    let cuboid = data.slice(s![.., y..y_max, x..x_max]).into_dyn();
                    polygon_set._add_tile(cuboid, (0, y, x));
                });
        });

    println!("Digesting polygon set ...");
    polygon_set.digest();
    assert_eq!(11049, polygon_set.len());
}
