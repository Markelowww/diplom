from tensorboard import program


logdir = "T:/OCTDL-main1111/neural/efficientvit_m4/log"


tb = program.TensorBoard()
tb.configure(argv=[None, "--logdir", logdir])
tb.main()
