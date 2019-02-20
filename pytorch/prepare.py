from pathlib import Path

img_dir = Path('./ccpd5000/train/')
img_paths = img_dir.glob('*.jpg')
img_paths = sorted(list(img_paths))

print(len(img_paths))

name = img_paths[0].name
print(name)

token = name.split('-')[3]
print(token)

token = token.replace('&', '_')
print(token)

values = token.split('_')
print(values)

values = [float(val) for val in values]
print(values)
