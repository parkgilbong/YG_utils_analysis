"""Snakemake rules for video processing utilities."""


rule flip_videos:
    input:
        video="{path}/{filename}.avi",
    output:
        video="{path}/{filename}_flipped.avi",
    log:
        "logs/flip/{path}/{filename}.log",
    run:
        from fp_behav.video.functions import flip_video
        flip_video(input.video, output.video)


rule resize_videos:
    input:
        video="{path}/{filename}.{ext}",
    output:
        video="{path}/{filename}_resized.{ext}",
    params:
        width=config.get("resize_width", 640),
        height=config.get("resize_height", 480),
    log:
        "logs/resize/{path}/{filename}.log",
    run:
        from fp_behav.video.functions import resize_video
        resize_video(input.video, output.video, params.width, params.height)
