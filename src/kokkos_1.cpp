
#include <Kokkos_Core.hpp>
#include <timer.hpp>

#include <iostream>
#include <random>

using real = double;
constexpr const int dim = 3;
// pointmass[0] is the mass, the next dim values are
// dimension coordinates, and the last ones are velocities
using pointmass_arr = double * [2 * dim + 1];
using pointmass = double[2 * dim + 1];

template <typename real>
struct SquareReduce {
  KOKKOS_INLINE_FUNCTION
  void operator()(const real i, real &lsum) const {
    lsum += i * i;
  }
};

void genBodies(int numPoints,
               typename Kokkos::View<
                   pointmass_arr>::HostMirror &bodies,
               bool fixedSeed = false) {
  using rngAlg = std::mt19937_64;
  std::random_device rd;
  rngAlg engine(rd());
  if(fixedSeed) {
    engine.seed(682185716);
  }
  std::uniform_real_distribution<real> genPos(-1.0, 1.0);
  std::uniform_real_distribution<real> genVel(-1.0, 1.0);
  std::uniform_real_distribution<real> genMass(
      1.52587890625e-5, std::numeric_limits<real>::max());
  for(int i = 0; i < numPoints; i++) {
    bodies(i, 0) = genMass(engine);
    for(int j = 1; j <= dim; j++) {
      bodies(i, j) = genPos(engine);
      bodies(i, j + dim) = genVel(engine);
    }
  }
}

struct force {
  real f[dim];
  void operator+=(const volatile force &other) volatile {
    for(int i = 0; i < dim; i++) {
      f[dim] += other.f[dim];
    }
  }
};

void computeForce(const int numPoints, const int curPoint,
                  const Kokkos::View<pointmass_arr> &bodies,
                  force &rF) {
  /* G in kg m^3 / s^2 */
  constexpr const real G = 6.674E-11;
  pointmass cur;
  cur[0] = bodies(curPoint, 0) * G;
  for(int i = 0; i < dim; i++) {
    rF.f[i] = 0.0;
    cur[i + 1] = bodies(curPoint, i + 1);
  }
  Kokkos::parallel_reduce(
      numPoints - 1,
      KOKKOS_LAMBDA(const int i, force &f) {
        int idx;
        if(i >= curPoint) {
          idx = i + 1;
        } else {
          idx = i;
        }
        real ds[dim];
        real dsNormSq = 0.0;
        real coeff = bodies(idx, 0) * cur[0];
        for(int j = 0; j < dim; j++) {
          ds[j] = bodies(idx, j + 1) - cur[j + 1];
          dsNormSq += ds[j] * ds[j];
          ds[j] *= coeff;
        }
        real dsInvNormCubed = 1.0 / sqrt(dsNormSq);
        for(int j = 0; j < 3; j++) {
          dsInvNormCubed *= dsInvNormCubed;
        }
        for(int j = 0; j < dim; j++) {
          f.f[j] += ds[j] * dsInvNormCubed;
        }
      },
      rF);
}

void NBodySim(int numPoints, real timestep, real maxTime) {
  Kokkos::View<pointmass_arr> bodies("Point Masses",
                                     numPoints);
  typename Kokkos::View<pointmass_arr>::HostMirror
      h_bodies = Kokkos::create_mirror_view(bodies);
  Kokkos::View<pointmass_arr> nextBodies("Point Masses",
                                         numPoints);
  genBodies(numPoints, h_bodies);
  for(real curTime = 0.0; curTime < maxTime;
      curTime += timestep) {
    Kokkos::parallel_for(
        numPoints, KOKKOS_LAMBDA(const int i) {
          force f;
          for(int j = 1; j <= dim; j++) {
            nextBodies(i, j) +=
                nextBodies(i, j + dim) * timestep;
            nextBodies(i, j + dim) += f.f[j - 1] * timestep;
          }
        });
    bodies = nextBodies;
  }
}

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  Timer::Timer t;
  constexpr const int n = 100000000;
  real sum = 0;
  t.startTimer();
  Kokkos::parallel_reduce(n, SquareReduce<real>(), sum);
  t.stopTimer();
  std::cout << "Sum of squares from 1 to " << n - 1 << ": "
            << sum << "\n";
  real check_sum = 0;
  t.startTimer();
  for(real i = 1; i < n; i++) {
    check_sum += i * i;
  }
  t.stopTimer();
  std::cout << "check_sum: " << check_sum << "\n";
  std::cout << t << "\nN Body Simulation Time:\n";
  t.reset();
  t.startTimer();
  NBodySim(5, 0.03125, 1.0);
  t.stopTimer();
  std::cout << t << "\n";
  Kokkos::finalize();
  return 0;
}
