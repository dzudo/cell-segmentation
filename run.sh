#!/bin/bash
for img in ../../dane_w3/png_avg/*.png
do
	#bacterial params
	python lset_seq.py -f $img -o ../res_w3_lset/ -r 250 -k 7 -t 100 -e 1.5 -s 5 -l 10.0 -a 0 -n 150
done

#for img in ../../dane_w1/png_avg/*.png
#do
#	#bacterial params
#	python lset_seq.py -f $img -o ../res_w1_lset/ -r 500 -k 7 -t 100 -e 1.5 -s 5 -l 10.0 -a -2 -n 100
#done
