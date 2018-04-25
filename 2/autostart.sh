#!/bin/bash

while :
do
	old_date=$(stat -c %Y main.cu)

	sleep 1

	current_date=$(stat -c %Y main.cu)
	if (( $current_date > $old_date )); then
		./run.sh
	fi
done