# Source with the dispatch directory and registered Fair-UK task.
root=$1
task=$2
draws=1000
case "$task" in
    stereo*) draws=0 ;;
esac
case "$task" in
    stereoset_en|winobias_en_controlled|winobias_en_natural|bbq_en)
        panel_input=(--dataset-bundle "$root/matched-en-v1/manifest.json")
        ;;
    stereoset_uk)
        panel_input=(--input "$root/../../ethos-domain-env-20261121/budget/uk-eval.csv")
        ;;
    winobias_uk_controlled|winobias_uk_natural|bbq_uk)
        panel_input=(--input "$root/../eval_sources/$task.csv")
        ;;
    *) return 1 ;;
esac
