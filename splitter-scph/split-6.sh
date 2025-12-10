#! /bin/bash

tlist="10 30 50 70 90 110 130 150 170 190 200 250 300 350 400 450 500 600 700 800 900 1000"
for t in $tlist; do
    name=6-run_${t}
    cp -f 6-calculate_effective_frequencies.py ${name}.py
    sed -i -e "s/^temperatures.*/temperatures=np.array([${t}])/" ${name}.py
    sed -i -e "s/^suffix_temperature.*/suffix_temperature=True/" ${name}.py

    cat > 6-run_${t}.sub <<EOF
#! /bin/bash
#SBATCH -J ${name}
#SBATCH -o ${name}.sout
#SBATCH -e ${name}.serr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH 

eval "\$(/opt/software/conda/bin/conda shell.bash hook)"
conda activate hiphive-1.4
cd $(pwd)
python ${name}.py >& ${name}.out

EOF
done
