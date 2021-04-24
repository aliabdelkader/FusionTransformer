# build image
```bash
	docker build -f Dockerfile -t fusion_image ..
```

# run image
```bash
	docker run -it -v `pwd`/../:/opt --runtime nvidia fusion_image
```
