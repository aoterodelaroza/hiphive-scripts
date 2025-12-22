#! /bin/bash

prefix=$(basename $(pwd))

nori=$(ls 6-run*py | wc -l)
nrun=$(ls ${prefix}-*.svib | wc -l)
ncon=$(grep Converged 6-run*out | wc -l)

if [ $nori -eq $nrun ] && [ $nori -eq $ncon ] ; then
    cat $(ls *.svib | sort -g | head -n 1) > ${prefix}.svib
    for file in $(ls *.svib | sort -g | tail -n+2) ; do
	tail -n 1 $file >> ${prefix}.svib
    done
else
    echo "some files missing: "
    echo "  py files: $nori"
    echo "  svib files: $nrun"
    echo "  converged: $ncon"
fi
