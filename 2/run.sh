#!/bin/bash
make all

./classsificator < args

python conv.py outtyp.data out.bmp

make cleaner
