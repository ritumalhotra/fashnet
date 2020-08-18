import torchvision.transforms as transforms
import albumentations

class AlbumentationImageDataSet(Dataset):
    def __init__(self, image_list):
        self.image_list = image_list

        self.aug = A.Compose([
            A.Resize(200, 300), 
            A.CenterCrop(100, 100), 
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p = 0.5), 
            A.Rotate(limit=(-90, 90)), 
            A.VerticalFlip(p=0.5), 
            A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return (len(self.image_list))

    
    def __getitem__(self, i):
        image = plt.imread(self.image_list[i])
        image = Image.fromarray(image).convert('RGB')
        image = self.aug(image=np.array(image))['image']
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return torch.tensor(image, dtype=torch.float32)