import argparse
import gala.potential as gp
import astropy.units as u
import pandas as pd
import cogsworth

def run_sim(porb_model="sana12", q_power_law=0, processes=128, extra_time=200 * u.Myr, file=None):
    pot = gp.load("/mnt/home/twagg/supernova-feedback/data/m11h_new_potential.yml")
    star_particles = pd.read_hdf("/mnt/home/twagg/supernova-feedback/data/FIRE_star_particles.h5", key="df")
    p = cogsworth.hydro.pop.HydroPopulation(star_particles=star_particles,
                                            max_ev_time=13736.52127883025 * u.Myr + extra_time,
                                            galactic_potential=pot,
                                            m1_cutoff=4,
                                            virial_parameter=1.0,
                                            subset=None,
                                            cluster_radius=3 * u.pc,
                                            cluster_mass=1e4 * u.Msun,
                                            processes=processes,
                                            sampling_params={"porb_model": porb_model,
                                                             "q_power_law": q_power_law})
    p.sample_initial_binaries()
    p.sample_initial_galaxy()

    print("Initial binaries and galaxy sampled, performing stellar evolution")

    p.perform_stellar_evolution()

    print("Stellar evolution complete, performing galactic evolution")

    p.perform_galactic_evolution()
    
    if p._initC is not None and "particle_id" not in p._initC.columns:
        p._initC["particle_id"] = p._initial_binaries["particle_id"]

    # save the results
    if file is None:
        file = f"feedback-sim-porb-{porb_model}-q-{q_power_law}"

    p.save(f"/mnt/home/twagg/ceph/pops/feedback-variations/{file}.h5", overwrite=True)


def main():
    parser = argparse.ArgumentParser(description='Supernova feedback simulation runner')
    parser.add_argument('-p', '--processes', default=128, type=int,
                        help='Number of processes to use')
    parser.add_argument('-t', '--extra_time', default=200, type=int,
                        help='Extra time to evolve for (in Myr)')
    parser.add_argument('-f', '--file', default=None, type=str,
                        help='Output file name')
    parser.add_argument('-P', '--porb_model', default="sana12", type=str,
                        help='Binary orbital period model')
    parser.add_argument('-q', '--q_power_law', default=0, type=int,
                        help='Binary mass ratio power law')
    args = parser.parse_args()

    # check if args.porb_model is a number and convert to dict if so
    try:
        args.porb_model = {
            "min": 0.15,
            "max": 5,
            "slope": float(args.porb_model)
        }
    except ValueError:
        pass

    run_sim(porb_model=args.porb_model,
            q_power_law=args.q_power_law,
            file=args.file,
            processes=args.processes,
            extra_time=args.extra_time * u.Myr)

if __name__ == "__main__":
    main()
