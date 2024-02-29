import pytest

from src.datamodules.components.deeplake_dataset import DeepLakeSegmentationDataset


@pytest.mark.parametrize('data_dir,stage,num_data', [
    ('hub://haritsahm/camvid_train', 'train', 369),
    ('hub://haritsahm/camvid_train', 'val', 100),
    ('hub://haritsahm/camvid_train', 'test', 232)
])
def test_load_dataset(data_dir, stage, num_data) -> None:
    ds = DeepLakeSegmentationDataset(data_dir, stage)

    assert len(ds) == num_data
