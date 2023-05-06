all:
	make compile
	make logs
compile: 
	python setup.py build_ext --inplace
logs:
	mkdir logs
clean:
	rm -r -f *.c *.so build
server:
	make compile
	make logs
	python fairnessAppServer.py --precompute --port ${PORT} --rgseed_test 9 
local:
	make compile
	make logs
	python fairnessAppServer.py --precompute --local --rgseed_test 9
experiment:
	sh run_all.sh