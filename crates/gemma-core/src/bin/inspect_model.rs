use gemma_core::model_store::ModelStore;
use gemma_io::io::Path as GemmaPath;

fn main() {
    let mut args = std::env::args().skip(1);
    let weights = args
        .next()
        .expect("usage: inspect_model <weights.sbs> [name_substr]");
    let name_filter = args.next();

    let store = ModelStore::new(GemmaPath::new(&weights));
    let mut mats = store.mat_ptrs().to_vec();
    mats.sort_by(|a, b| a.name().cmp(b.name()));

    for m in mats {
        if let Some(f) = &name_filter {
            if !m.name().contains(f) {
                continue;
            }
        }
        println!(
            "{}\t{:?}\tscale={}\trows={}\tcols={}\tpacked_bytes={}",
            m.name(),
            m.ty(),
            m.scale(),
            m.rows(),
            m.cols(),
            m.packed_bytes()
        );
    }
}
