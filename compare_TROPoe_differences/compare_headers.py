import netCDF4 as nc
import sys

def save_header(filepath, outfile):
    ds = nc.Dataset(filepath)
    with open(outfile, "w") as f:
        # Redirect stdout temporarily
        old_stdout = sys.stdout
        sys.stdout = f
        print("=== DIMENSIONS ===")
        for dim, val in ds.dimensions.items():
            print(f"  {dim}: {len(val)}")
        print("\n=== VARIABLES ===")
        for var in ds.variables:
            v = ds.variables[var]
            print(f"  {var} {v.dimensions} {v.dtype}")
            for attr in v.ncattrs():
                print(f"    {attr}: {getattr(v, attr)}")
        print("\n=== GLOBAL ATTRIBUTES ===")
        for attr in ds.ncattrs():
            print(f"  {attr}: {getattr(ds, attr)}")
        sys.stdout = old_stdout
    ds.close()

save_header("C:/Users/c7071147/Documents/TROPoe_run/dave_innit/tropoe/hatpro/tropoe.c1.20251201.000015.nc", 'my_header.txt')
print('end')
save_header("I:/User/Documents/Research/Running_TROPoe/Download from Dave/beth_saunders/tropoe/hatpro/tropoe.c1.20251201.000015.nc", 'dave_header.txt')
print('end')