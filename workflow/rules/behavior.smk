"""Snakemake rules for behavioral video preprocessing and DLC conversion."""


rule behav_preprocess:
    input:
        frames_dir=lambda wc: f"{config['raw_video_path']}/{wc.animal}/{wc.session}",
    output:
        video="results/{animal}/{session}/behavior/video.mp4",
        flag="results/{animal}/{session}/behavior/done.flag",
    log:
        "logs/behav_preprocess/{animal}_{session}.log",
    run:
        from fp_behav.behavior.preprocessing import preprocess_behavioral_data
        preprocess_behavioral_data(config_path="configs/behavior.yaml")
        open(output.flag, "w").close()


rule dlc2boris:
    input:
        dlc_csv=lambda wc: f"results/{wc.animal}/{wc.session}/behavior/{wc.animal}_DLC.csv",
    output:
        boris_csv="results/{animal}/{session}/behavior/{animal}_BORIS.csv",
    params:
        variant=config.get("dlc2boris_variant", "base"),
    log:
        "logs/dlc2boris/{animal}_{session}.log",
    run:
        if params.variant == "ct":
            from fp_behav.behavior.boris.ct import main
        elif params.variant == "di":
            from fp_behav.behavior.boris.di import main
        else:
            from fp_behav.behavior.boris.base import main
        main()
