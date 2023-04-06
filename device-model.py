# helper functions for device
def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get GPU device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids
  
# example of usage
n_gpu = 2
device, device_ids = prepare_device(n_gpu)

# other related code
device = 'cuda' if torch.cuda.is_available() else 'cpu'




model_path = "model.pth"

# save model
if isinstance(model, (nn.DataParallel, nn.DistributedDataParallel)):
    torch.save(model.module.state_dict(), model_path)
else:
    torch.save(model.state_dict(), model_path)
    
# load model
model = nn.DataParallel(model, **gpu_device_arg) # multi-gpu model
if isinstance(model, (nn.DataParallel, nn.DistributedDataParallel)):
    model.module.load_state_dict(model_path)    # your model will be loaded to multi-gpu model.
else:
    model.load_state_dict(model_path)
    
# save model single GPU
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optim.state_dict(),
    'train_loss': train_loss,
}, model_path)
 
