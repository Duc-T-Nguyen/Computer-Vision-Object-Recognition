import scipy.io
from two_pass_hoi import HICODetDataset

mat = scipy.io.loadmat(
    'hico_20160224_det/anno_bbox.mat',
    struct_as_record=False,
    squeeze_me=True
)

# Print every throw-related HOI category with its actual 1-indexed id
print("All throw-related HOI categories in your .mat file:")
for i, action in enumerate(mat['list_action']):
    name = f"{action.vname}_{action.nname}"
    if 'throw' in name.lower():
        print(f"  id={i+1:3d}  {name}")

# Then check how many samples got labeled as throws with your current ids
dataset = HICODetDataset('hico_20160224_det', split='train')
throw_samples = [s for s in dataset.samples if s['label'] == 1.0]
print(f"\nThrow samples found with current THROW_HOI_IDS={HICODetDataset.THROW_HOI_IDS}: {len(throw_samples)}")


dataset = HICODetDataset('hico_20160224_det', split='train')
train_labels = [s['label'] for s in dataset.samples]
n_throw    = sum(train_labels)
n_nonthrow = len(train_labels) - n_throw
print(f"Throw    : {int(n_throw)}")
print(f"Non-throw: {int(n_nonthrow)}")
print(f"Samples per epoch with current sampler: {int(2 * n_throw)}")
print(f"Batches per epoch at batch_size=32: {int(2 * n_throw) // 32}")
