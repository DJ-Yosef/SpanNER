# encoding: utf-8
import os
from pytorch_lightning import Trainer
from models.tagger import BertNerTagger
def evaluate(ckpt, hparams_file):
	"""main"""

	trainer = Trainer(gpus=[5], distributed_backend="dp")
	# trainer = Trainer(distributed_backend="dp")

	model = BertNerTagger.load_from_checkpoint(
		checkpoint_path=ckpt,
		hparams_file=hparams_file,
		map_location=None,
		batch_size=1,
		max_length=128,
		workers=0
	)
	trainer.test(model=model)


if __name__ == '__main__':

	root_dir1 = "/home/jlfu/SPred/train_logs"

	midpath = "notetc/spanPred_bert-large-uncased_prunFalse_spLenTrue_spMorphFalse_SpWtFalse_value1_52887159"
	model_names = ["epoch=9.ckpt"]
	for mn in model_names:
		print("model-name: ", mn)
		CHECKPOINTS = "/home/jlfu/SPred/train_logs/" + midpath + "/" + mn
		HPARAMS = "/home/jlfu/SPred/train_logs/" + midpath + "/lightning_logs/version_0/hparams_prune.yaml"
		evaluate(ckpt=CHECKPOINTS, hparams_file=HPARAMS)



