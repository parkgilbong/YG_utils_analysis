"""Snakemake rules for Fiber Photometry preprocessing and analysis."""


rule fp_preprocess:
    input:
        tank=lambda wc: f"{config['raw_data_path']}/{config['recorded_date']}/{wc.animal}",
    output:
        flag="results/{animal}/{session}/Preprocessing/done.flag",
        report="results/{animal}/{session}/Preprocessing/report.pdf",
    params:
        dest=lambda wc: f"results/{wc.animal}/{wc.session}/Preprocessing",
        sys=config.get("sys", "tdt"),
        detrending=config.get("detrending_method", "Highpass_filter"),
        fps=config.get("fps", 25),
        rec_duration=config.get("rec_duration", 600),
    log:
        "logs/fp_preprocess/{animal}_{session}.log",
    run:
        import logging
        from fp_behav.fp import functions as FPFunctions
        from fp_behav.io import report as ReportGeneration

        logging.basicConfig(filename=log[0], level=logging.INFO)
        try:
            FPFunctions.FP_preprocessing_1ch(
                Tank_path=input.tank,
                Dest_folder=params.dest,
                sys=params.sys,
                Detrending_method=params.detrending,
                FPS=params.fps,
                Rec_duration=params.rec_duration,
            )
            image_paths = [
                f"{params.dest}/Plot_Raw_signal_465.png",
                f"{params.dest}/Plot_Denoised_signals.png",
                f"{params.dest}/Plot_405_465_correlation.png",
                f"{params.dest}/Plot_465_z-score.png",
            ]
            ReportGeneration.FP_preprocessing(
                output_path=f"results/{wildcards.animal}/{wildcards.session}",
                title=f"{wildcards.animal}_{wildcards.session}",
                image_paths=image_paths,
                comments=["Raw signals", "Denoised signals", "405-465 correlation", "465 zscore"],
            )
            open(output.flag, "w").close()
        except Exception as e:
            logging.error(str(e))
            raise


rule fp_epoch:
    input:
        scoring=lambda wc: f"results/{wc.animal}/{wc.session}/scoring.tsv",
        preprocessed=lambda wc: f"results/{wc.animal}/{wc.session}/Preprocessing/done.flag",
    output:
        "results/{animal}/{session}/epoch/done.flag",
    log:
        "logs/fp_epoch/{animal}_{session}.log",
    run:
        from fp_behav.fp.epoch import run_epoch_analysis
        run_epoch_analysis(config, project_root=".")
        open(output[0], "w").close()


rule fp_peak:
    input:
        "results/{animal}/{session}/Preprocessing/done.flag",
    output:
        "results/{animal}/{session}/peak/done.flag",
    log:
        "logs/fp_peak/{animal}_{session}.log",
    run:
        from fp_behav.fp.peak import run_peak_analysis
        run_peak_analysis(config)
        open(output[0], "w").close()
