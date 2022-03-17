#!/bin/bash
FILE=$1

for i in `seq 0 250`
do
    echo	v${i}	$(	grep -w	v${i}	$FILE	| wc -l	)
done
