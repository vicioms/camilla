import zarr, fsspec, os

url = "https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr"
src = fsspec.get_mapper(url)  # HTTP-backed Zarr store

store_dir = zarr.DirectoryStore("/Users/schimmenti/Desktop/DresdenProjects/camilla/folderer/ZSNS001.ome.zarr")
group_dir  = zarr.group(store = store_dir, overwrite=True)

#zarr.copy_store(src, dst, log=print)  # mirror the whole OME-Zarr
zarr.convenience.copy_all(src, group_dir, if_exists='replace', dry_run=False, log=print)  # mirror the whole OME-Zarr