import os, sys, glob, time, pathlib, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

# Kerasa / TensorFlow
from loss import depth_loss_function
from utils import predict, save_images, load_test_data, load_void_test_data
from model import create_model, create_two_branch_model, create_two_branch_model_very_late_fusion
from model_resnet import create_model_resnet
from data import get_nyu_train_test_data, get_unreal_train_test_data, get_void_train_test_data
from callbacks import get_nyu_callbacks, get_void_callbacks
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--data', default='nyu', type=str, help='Training dataset.')
parser.add_argument('--voidmode', default='normal', type=str, help='VOID training mode.')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--bs', type=int, default=4, help='Batch size')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
parser.add_argument('--gpus', type=int, default=1, help='The number of GPUs to use')
parser.add_argument('--gpuids', type=str, default='0', help='IDs of GPUs to use')
parser.add_argument('--mindepth', type=float, default=10.0, help='Minimum of input depths')
parser.add_argument('--maxdepth', type=float, default=1000.0, help='Maximum of input depths')
parser.add_argument('--name', type=str, default='densedepth_nyu', help='A name to attach to the training session')
parser.add_argument('--checkpoint', type=str, default='', help='Start training from an existing model.')
parser.add_argument('--weights', type=str, default='', help='Start training with pretrained weights.')
parser.add_argument('--full', dest='full', action='store_true', help='Full training with metrics, checkpoints, and image samples.')
parser.add_argument('--resnet50', dest='resnet50', action='store_true', help='Train a Resnet 50 model.')
parser.add_argument('--dont-interpolate', default=False, dest='dont_interpolate', action='store_true', help='Use raw sparse depth maps.')
parser.add_argument('--channels', type=int, default=3, help='Channels')
parser.add_argument('--use-void-1500', default=False, dest='use_void_1500', action='store_true', help='Use VOID 1500 raw sparse depth maps.')
parser.add_argument('--use-weigted-early-fusion', default=False, dest='use_weigted_early_fusion', action='store_true', help='Use weighted early fusion.')
parser.add_argument('--use-very-late-fusion', default=False, dest='use_very_late_fusion', action='store_true', help='Concat branches at the very end (just before last conv layer).')

args = parser.parse_args()

# Inform about multi-gpu training
if args.gpus == 1: 
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuids
    print('Will use GPU ' + args.gpuids)
else:
    print('Will use ' + str(args.gpus) + ' GPUs.')

channels = args.channels
# Create the model
if args.data == 'void' and args.voidmode == 'two-branch':
    if args.use_very_late_fusion: model = create_two_branch_model_very_late_fusion( existing=args.checkpoint, channels=channels)
    else: model = create_two_branch_model( existing=args.checkpoint, channels=channels)
elif args.data == 'void' and args.voidmode == '5channel':
    model = create_model(existing=args.checkpoint, channels=5)
    channels = 5
elif args.data == 'void' and args.voidmode == '4channel':
    if args.use_weigted_early_fusion: model = create_model_early(existing=args.checkpoint, channels=4)
    else: model = create_model(existing=args.checkpoint, channels=4)
    channels = 4
elif args.resnet50:  # if want a resnet model
    model = create_model_resnet(existing=args.checkpoint)
else:
    model = create_model( existing=args.checkpoint)
if args.weights != '':
    model.load_weights(args.weights)
# Data loaders
if args.data == 'nyu': train_generator, test_generator = get_nyu_train_test_data( args.bs )
if args.data == 'unreal': train_generator, test_generator = get_unreal_train_test_data( args.bs )
if args.data == 'void': train_generator, test_generator = get_void_train_test_data( args.bs, mode=args.voidmode, dont_interpolate=args.dont_interpolate, channels=channels, use_void_1500=args.use_void_1500 )
# Training session details
runID = str(int(time.time())) + '-n' + str(len(train_generator)) + '-e' + str(args.epochs) + '-bs' + str(args.bs) + '-lr' + str(args.lr) + '-' + args.name
outputPath = '/home/mikkel/models/'
runPath = outputPath + runID
pathlib.Path(runPath).mkdir(parents=True, exist_ok=True)
print('Output: ' + runPath)

 # (optional steps)
if True:
    # Keep a copy of this training script and calling arguments
    with open(__file__, 'r') as training_script: training_script_content = training_script.read()
    training_script_content = '#' + str(sys.argv) + '\n' + training_script_content
    with open(runPath+'/'+__file__, 'w') as training_script: training_script.write(training_script_content)

    # Generate model plot
    plot_model(model, to_file=runPath+'/model_plot.svg', show_shapes=True, show_layer_names=True)

    # Save model summary to file
    from contextlib import redirect_stdout
    with open(runPath+'/model_summary.txt', 'w') as f:
        with redirect_stdout(f): model.summary()

# Multi-gpu setup:
basemodel = model
if args.gpus > 1: model = multi_gpu_model(model, gpus=args.gpus)

# Optimizer
optimizer = Adam(lr=args.lr, amsgrad=True)

# Compile the model
print('\n\n\n', 'Compiling model..', runID, '\n\n\tGPU ' + (str(args.gpus)+' gpus' if args.gpus > 1 else args.gpuids)
        + '\t\tBatch size [ ' + str(args.bs) + ' ] ' + ' \n\n')
model.compile(loss=depth_loss_function, optimizer=optimizer)

print('Ready for training!\n')

# Callbacks
callbacks = []
if args.data == 'nyu': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)
if args.data == 'unreal': callbacks = get_nyu_callbacks(model, basemodel, train_generator, test_generator, load_test_data() if args.full else None , runPath)
if args.data == 'void': callbacks = get_void_callbacks(model, basemodel, train_generator, test_generator, load_void_test_data(channels=channels, dont_interpolate=args.dont_interpolate, use_void_1500=args.use_void_1500) if args.full else None , runPath)
if args.data == 'void-imu': callbacks = get_void_callbacks(model, basemodel, train_generator, test_generator, None , runPath)

# Start training
model.fit_generator(train_generator, callbacks=callbacks, validation_data=test_generator, epochs=args.epochs, shuffle=True)

# Save the final trained model:
basemodel.save(runPath + '/model.h5')
