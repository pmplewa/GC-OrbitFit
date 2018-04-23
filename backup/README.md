## Compile & Run

* [REBOUND](https://github.com/hannorein/rebound)
* [REBOUNDx](https://github.com/dtamayo/reboundx)
* [MultiNest](https://github.com/farhanferoz/MultiNest)

```
cd rebound
make

cd ../reboundx
make

cd ../MultiNest/build
cmake ..
make

cd ../..
ln -s rebound/librebound.so
ln -s reboundx/libreboundx.so
ln -s MultiNest/lib/libmultinest_mpi.so

make
mkdir chains

mpirun -n 8 sample
```
