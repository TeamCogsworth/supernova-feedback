import gala.potential as gp
import astropy.units as u
import pandas as pd
import cogsworth

def main():
    
    m11h = cogsworth.pop.load("/mnt/home/twagg/ceph/pops/m11h-r-3-no-gal")
    
    m11h.initial_galaxy._Z = m11h.initial_galaxy._Z / 2
    m11h._initial_binaries = None
    m11h._initC["metallicity"] = m11h._initC["metallicity"] / 2
    m11h._orbits_file = None
    m11h._orbits = []
    m11h._final_pos = None
    m11h._final_vel = None
    
    m11h.perform_stellar_evolution()
    
    m11h.save(f"/mnt/home/twagg/ceph/pops/m11h-r-3-Z-0.5-same-initC", overwrite=True)

if __name__ == "__main__":
    main()
