from datasets.mvtec import MVTEC


# =========================
# Train
# =========================

train_dataset = MVTEC(
    root="./data/weld",
    class_name="weld",
    train=True,
    normalize="imagenet",
    img_size=224,
    crp_size=224,
    msk_size=224,
    msk_crp_size=224,
)

print("Train samples:", len(train_dataset))

img, label, mask, class_name = train_dataset[0]

print("Train")
print("image :", img.shape)
print("label :", label)
print("mask  :", mask.shape)
print("class :", class_name)


# =========================
# Test
# =========================

test_dataset = MVTEC(
    root="./data/weld",
    class_name="weld",
    train=False,
    normalize="imagenet",
    img_size=224,
    crp_size=224,
    msk_size=224,
    msk_crp_size=224,
)

print()
print("Test samples:", len(test_dataset))

for i in range(len(test_dataset)):

    img, label, mask, class_name = test_dataset[i]

    print(
        i,
        img.shape,
        label,
        mask.shape,
        mask.max().item(),
        class_name,
    )