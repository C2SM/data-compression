import xarray as xr
import zarr

# Specify the path to your zipped Zarr file
zarr_file = 'remap_umfl_s_20220728T000000Z.nc.=.field_umfl_s.=.rank_0.zarr.zip'

# Open the Zarr store
# The 'field_umfl_s' part of the filename suggests this is the data variable.
try:
    # xarray can automatically handle Zarr stores within a zip archive.
    # We specify the engine as 'zarr' and tell it to use the `fsspec` library
    # to open the zip file.
    ds = xr.open_zarr(zarr_file, engine='zarr')

    # Access the 'field_umfl_s' variable
    data_array = ds['field_umfl_s']

    # Convert the xarray DataArray to a NumPy array
    numpy_array = data_array.values

    print("Conversion successful! Here is the shape of the numpy array:")
    print(numpy_array.shape)

except FileNotFoundError:
    print(f"Error: The file '{zarr_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
