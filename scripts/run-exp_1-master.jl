const SCRIPTS_DIR = "/Users/kai/projects/a-reproducible-research-workflow/scripts"

# NOTE Threads.@threads doesn't support nested loop.
Threads.@threads for (lr, bs) in [(lr, bs) for lr in [1e-3, 1e-2, 1e-1, 1e-0], bs in 20:20:100]
    cmd = `python $SCRIPTS_DIR/run-exp_1.py --lr=$lr --batch-size=$bs`
    @info "Running" cmd
    run(cmd)
end
