extern crate flatc_rust;

use std::path::Path;

fn main() {
    // println!("cargo:rerun-if-changed=flatbuffers/common.fbs");
    // println!("cargo:rerun-if-changed=flatbuffers/delta.fbs");
    // println!("cargo:rerun-if-changed=flatbuffers/hierarchy.fbs");
    // println!("cargo:rerun-if-changed=flatbuffers/manifest.fbs");
    // println!("cargo:rerun-if-changed=flatbuffers/tile.fbs");
    // println!("cargo:rerun-if-changed=flatbuffers/transaction.fbs");
    // flatc_rust::run(flatc_rust::Args {
    //     out_dir: Path::new("src/generated"),
    //     inputs: &[
    //         Path::new("flatbuffers/common.fbs"),
    //         Path::new("flatbuffers/delta.fbs"),
    //         Path::new("flatbuffers/hierarchy.fbs"),
    //         Path::new("flatbuffers/manifest.fbs"),
    //         Path::new("flatbuffers/tile.fbs"),
    //         Path::new("flatbuffers/transaction.fbs"),
    //     ],
    //     ..Default::default()
    // })
    // .expect("flatc-rust");
}
