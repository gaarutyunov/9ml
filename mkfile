</$objtype/mkfile

all:V: src tests

src:V:
	cd src && mk

tests:V:
	cd src/tests && mk

clean:V:
	cd src && mk clean
	cd src/tests && mk clean
