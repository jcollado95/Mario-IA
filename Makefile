SHELL=/bin/bash

install:
	@echo "Installing libraries..."
	pip install -r requirements.txt
	@echo "Setting up the project..."
	cp ${VIRTUAL_ENV}/lib/python3.6/site-packages/gym_super_mario_bros/_roms/*.nes ${VIRTUAL_ENV}/lib/python3.6/site-packages/retro/data/stable/SuperMarioBros-Nes
	mv ${VIRTUAL_ENV}/lib/python3.6/site-packages/retro/data/stable/SuperMarioBros-Nes/{super-mario-bros,rom}.nes 
	@echo "Project ready."
