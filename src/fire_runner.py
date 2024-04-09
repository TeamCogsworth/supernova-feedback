import argparse
import gala.potential as gp
import astropy.units as u
import pandas as pd
import cogsworth

def run_sim(alpha_vir, alpha_ce, mt_eff=-1, ecsn_kick=-20, bhflag=1, subset=None, processes=32, extra_time=200 * u.Myr):
    pot = gp.load("/mnt/home/twagg/supernova-feedback/data/m11h_potential.yml")
    star_particles = pd.read_hdf("/mnt/home/twagg/supernova-feedback/data/FIRE_star_particles.h5", key="df")
    p = cogsworth.hydro.pop.HydroPopulation(star_particles=star_particles,
                                            max_ev_time=13736.52127883025 * u.Myr + extra_time,
                                            galactic_potential=pot,
                                            m1_cutoff=4,
                                            virial_parameter=alpha_vir,
                                            subset=subset,
                                            cluster_radius=1 * u.pc,
                                            cluster_mass=1e4 * u.Msun,
                                            processes=processes,
                                            BSE_settings={"alpha1": alpha_ce, "acc_lim": mt_eff,
                                                          'bhflag': bhflag, 'sigmadiv': ecsn_kick})
    p.create_population()
    
    if p._initC is not None and "particle_id" not in p._initC.columns:
        p._initC["particle_id"] = p._initial_binaries["particle_id"]

    # save the results
    p.save(f"/mnt/home/twagg/ceph/pops/m11h-r-3", overwrite=True)


def main():
    parser = argparse.ArgumentParser(description='Boundedness simulation runner')
    parser.add_argument('-a', '--alpha_vir', default=1.0, type=float,
                        help='Star particle virial parameter')
    parser.add_argument('-c', '--alpha_ce', default=1.0, type=float,
                        help='Common-envelope efficiency')
    parser.add_argument('-b', '--beta', default=-1, type=float,
                        help='Mass transfer efficiency')
    parser.add_argument('-k', '--bhflag', default=1, type=int,
                        help='BH kick flag')
    parser.add_argument('-e', '--ecsn-kick', default=-20, type=int,
                        help='ECSN kick strength')
    parser.add_argument('-s', '--subset', default=None, type=int,
                        help='Size of subset of star particles to use')
    parser.add_argument('-p', '--processes', default=32, type=int,
                        help='Number of processes to use')
    parser.add_argument('-t', '--extra_time', default=200, type=int,
                        help='Extra time to evolve for (in Myr)')
    args = parser.parse_args()

    run_sim(alpha_vir=args.alpha_vir, alpha_ce=args.alpha_ce, mt_eff=args.beta, bhflag=args.bhflag,
            ecsn_kick=args.ecsn_kick, subset=args.subset, processes=args.processes,
            extra_time=args.extra_time * u.Myr)

if __name__ == "__main__":
    main()
