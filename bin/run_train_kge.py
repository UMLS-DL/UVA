import os
import pickle
from absl import app, flags, logging

import torch
from torch.optim import Adam, SGD

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss
from pykeen.training import SLCWATrainingLoop
from pykeen.trackers import CSVResultTracker
from pykeen.models import TransE, TransR, DistMult, HolE, RESCAL, ComplEx, RotatE, ConvKB

FLAGS = flags.FLAGS

# Type of Models
flags.DEFINE_boolean("use_self_entity_relation_ids_map", True,
                     "Specify whether to use self-generated entity2id & relation2id files")
flags.DEFINE_string("kge_model", "TransE",
                    "Specify the KGE variant (TransE, TransR, DistMult, ComplEx, RESCAL, ConvE, ConKB, RGCN)")
flags.DEFINE_string("kge_triple_variants", "All_Triples",
                    "Specify the KGE variant (All_Triples, AUI_SCUI_Triples, SCUI_ParentSCUI_Triples, SCUI_SG_Triples)")

# Model Hyperparameters
flags.DEFINE_integer("random_seed", 3338,
                     "Specify the random seed to keep reproducibility")
flags.DEFINE_boolean("create_inverse_triples", False,
                     "Specify create_inverse_triples True or False")
flags.DEFINE_integer("embedding_dim", 50,
                     "Specify the embedding dimension")
flags.DEFINE_string("optimizer", "Adam",
                    "Specify the type of optimizer (SGD, Adam)")
flags.DEFINE_float("lr", 0.01,
                   "Specify the learning rate")
flags.DEFINE_string("loss", "marginranking",
                    "Specify the type of loss function")
flags.DEFINE_integer("margin", 1,
                     "Specify the margin")
flags.DEFINE_string("training_loop", "slcwa",
                    "Specify the training loop")
flags.DEFINE_integer("num_epochs", 100,
                     "Specify the number of training epochs")
flags.DEFINE_integer("batch_size", 1024,
                     "Specify the number of batches")
flags.DEFINE_string("evaluator", "rankbased",
                    "Specify the number of batches")
flags.DEFINE_boolean("enable_eval_filtering", True,
                     "Specify whether to enable_eval_filtering")
flags.DEFINE_string("negative_sampler", "basic",
                    "Specify the negative sampler")
flags.DEFINE_integer("num_negs_per_pos", 50,
                     "Specify the ratio of negative entity sampling")
flags.DEFINE_string("checkpoint_name", "my_checkpoint.pt",
                    "Specify the checkpoint name during training")
flags.DEFINE_integer("checkpoint_frequency", 20,
                    "Specify the checkpoint saving interval in minutes")
flags.DEFINE_string("device", "cuda:0",
                    "Specify the device (cuda:0 or cpu)")

# I/O Directory Locations
flags.DEFINE_string("root_dir", "..",
                    "Specify the root location")
flags.DEFINE_string("input_data_dir", "2-BYOD_UMLS",
                    "Specify the location of input data")
flags.DEFINE_string("result_logs_out_dir", "5-Logs/1-pykeen_train_kge",
                    "Specify the location of training result logs")
flags.DEFINE_string("checkpoint_dir", "6-Outputs/checkpoints",
                    "Specify the checkpoint location during training")
flags.DEFINE_string("models_out_dir", "6-Outputs/models",
                    "Specify the export location for the trained model")
flags.DEFINE_string("embeddings_out_dir", "6-Outputs/embeddings",
                    "Specify the export location for the trained embeddings")


def main(_):
    print("\n======== Specifications: ========\n")
    print("Model: {}".format(FLAGS.kge_model))
    print("Triple Variant: {}".format(FLAGS.kge_triple_variants))
    print("Embedding Dimension: {}".format(FLAGS.embedding_dim))
    print("Optimizer: {}".format(FLAGS.optimizer))
    print("Learning Rate: {}".format(FLAGS.lr))
    print("Loss Function: {}".format(FLAGS.loss))
    print("Loss Margin: {}".format(FLAGS.margin))
    print("Training Loop: {}".format(FLAGS.training_loop))
    print("Number of Epochs: {}".format(FLAGS.num_epochs))
    print("Batch Size: {}".format(FLAGS.batch_size))
    print("Negative Sampler: {}".format(FLAGS.negative_sampler))
    print("Negative Entity Sampling Ratio: {}".format(FLAGS.num_negs_per_pos))

    # Import BYOD_UMLS dataset variant
    BYOD_UMLS_TRAIN_PATH = "{}/{}/train2id.txt".format(
        FLAGS.root_dir,
#        FLAGS.input_data_dir, 
        FLAGS.kge_triple_variants
    )

    # Construct the result logs, checkpoints, trained model and embedding
    # saving path
    result_logs_output_PATH = "{}/{}/{}_{}_{}/{}/logs.txt".format(
        FLAGS.root_dir,
        FLAGS.result_logs_out_dir,
        FLAGS.kge_model,
        FLAGS.optimizer,
        FLAGS.num_epochs,
        FLAGS.kge_triple_variants
    )

    checkpoint_output_PATH = "{}/{}/{}_{}_{}/{}".format(
        FLAGS.root_dir,
        FLAGS.checkpoint_dir,
        FLAGS.kge_model,
        FLAGS.optimizer,
        FLAGS.num_epochs,
        FLAGS.kge_triple_variants
    )

    model_output_PATH = "{}/{}/{}_{}_{}/{}".format(
        FLAGS.root_dir,
        FLAGS.models_out_dir,
        FLAGS.kge_model,
        FLAGS.optimizer,
        FLAGS.num_epochs,
        FLAGS.kge_triple_variants
    )

    embedding_output_PATH = "{}/{}/{}_{}_{}/{}".format(
        FLAGS.root_dir,
        FLAGS.embeddings_out_dir,
        FLAGS.kge_model,
        FLAGS.optimizer,
        FLAGS.num_epochs,
        FLAGS.kge_triple_variants
    )

    # Create directories
    if not os.path.exists("{}".format(checkpoint_output_PATH)):
        os.makedirs("{}".format(checkpoint_output_PATH))
    if not os.path.exists("{}".format(model_output_PATH)):
        os.makedirs("{}".format(model_output_PATH))
    if not os.path.exists("{}".format(embedding_output_PATH)):
        os.makedirs("{}".format(embedding_output_PATH))

    print("Training:      {}".format(BYOD_UMLS_TRAIN_PATH))
    print("Logs:          {}".format(result_logs_output_PATH))
    print("Checkpoint:    {}".format(checkpoint_output_PATH))
    print("Model Out:     {}".format(model_output_PATH))
    print("Embedding Out: {}\n".format(embedding_output_PATH))

    if FLAGS.use_self_entity_relation_ids_map:
        BYOD_UMLS_ENTITY_TO_ID = "{}/{}/entity2id.txt".format(
            FLAGS.root_dir, FLAGS.kge_triple_variants)
        BYOD_UMLS_RELATION_TO_ID = "{}/{}/relation2id.txt".format(
            FLAGS.root_dir, FLAGS.kge_triple_variants)

        # Generate ENTITY_TO_ID and RELATION_TO_ID mappings
        ENTITY_TO_ID = {}
        RELATION_TO_ID = {}

        print("Using self entity2id.txt to generate ENTITY_TO_ID mapping...")
        with open(BYOD_UMLS_ENTITY_TO_ID, "r") as f:
            lines = f.readlines()
            for line in lines:
                entity_name, entity_id = line.split("\t")
                ENTITY_TO_ID[entity_name] = entity_id
        print("Done!")

        print("Using self relation2id.txt to generate RELATION_TO_ID mapping...")
        with open(BYOD_UMLS_RELATION_TO_ID, "r") as f:
            lines = f.readlines()
            for line in lines:
                relation_name, relation_id = line.split("\t")
                RELATION_TO_ID[relation_name] = relation_id
        print("Done!")

        print("\nStarting KGE training...")
        # Training Pipeline
        training = TriplesFactory.from_path(
            path=BYOD_UMLS_TRAIN_PATH,
            entity_to_id=ENTITY_TO_ID,
            relation_to_id=RELATION_TO_ID,
            create_inverse_triples=FLAGS.create_inverse_triples
        )
    else:
        print("\nStarting KGE training...")
        # Training Pipeline
        training = TriplesFactory.from_path(
            path=BYOD_UMLS_TRAIN_PATH,
            create_inverse_triples=FLAGS.create_inverse_triples
        )
    
    print("Saving PyKeen-generated ENTITY_TO_ID map...")
    pickle.dump(training.entity_to_id,
                open(
                    "{}/entity2id_dump.p".format(embedding_output_PATH), "wb"),
                protocol=4)
    print("{}/entity2id_dump.p write success!\n".format(embedding_output_PATH))

    print("Saving PyKeen-generated RELATION_TO_ID map...")
    pickle.dump(training.relation_to_id,
                open(
                    "{}/relation2id_dump.p".format(embedding_output_PATH), "wb"),
                protocol=4)
    print("{}/relation2id_dump.p write success!\n".format(embedding_output_PATH))

    ##### LOSS #####
    loss_kwargs = {
        "margin": FLAGS.margin
    }

    loss = MarginRankingLoss(**loss_kwargs)

    ##### MODEL #####
    model_kwargs = {
        "triples_factory": training,
        "embedding_dim": FLAGS.embedding_dim,
        "loss": loss,
        "random_seed": FLAGS.random_seed,
        "preferred_device": FLAGS.device
    }

    if FLAGS.kge_model == "TransE":
        model = TransE(**model_kwargs)
    elif FLAGS.kge_model == "TransR":
        model = TransR(**model_kwargs)
    elif FLAGS.kge_model == "DistMult":
        model = DistMult(**model_kwargs)
    elif FLAGS.kge_model == "HolE":
        model = HolE(**model_kwargs)
    elif FLAGS.kge_model == "RESCAL":
        model = RESCAL(**model_kwargs)
    elif FLAGS.kge_model == "ComplEx":
        model = ComplEx(**model_kwargs)
    elif FLAGS.kge_model == "RotatE":
        model = RotatE(**model_kwargs)
    elif FLAGS.kge_model == "ConvKB":
        model = ConvKB(**model_kwargs)

    ##### OPTIMIZER #####
    optimizer_kwargs = {
        "params": model.get_grad_params(),
        "lr": FLAGS.lr
    }

    if FLAGS.optimizer == "SGD":
        optimizer = SGD(**optimizer_kwargs)
    elif FLAGS.optimizer == "Adam":
        optimizer = Adam(**optimizer_kwargs)
    
    ##### NEGATIVE SAMPLER #####
    negative_sampler_kwargs = dict(
        num_negs_per_pos = FLAGS.num_negs_per_pos
    )

    ##### TRAINING SETUP #####
    training_kwargs = {
        "model": model,
        "triples_factory": training,
        "optimizer": optimizer,
        "negative_sampler": FLAGS.negative_sampler,
        "negative_sampler_kwargs": negative_sampler_kwargs
    }

    training_loop = SLCWATrainingLoop(**training_kwargs)

    ##### RESULT TRACKER #####
    result_tracker_kwargs = {
        "path": result_logs_output_PATH
    }
    
    result_tracker = CSVResultTracker(**result_tracker_kwargs)

    ##### BEGIN TRAINING #####
    training_loop_kwargs = {
        "triples_factory": training,
        "num_epochs": FLAGS.num_epochs,
        "batch_size": FLAGS.batch_size,
        "checkpoint_name": FLAGS.checkpoint_name,
        "checkpoint_frequency": FLAGS.checkpoint_frequency,
        "checkpoint_directory": checkpoint_output_PATH,
        "result_tracker": result_tracker
    }

    training_loss = training_loop.train(**training_loop_kwargs)

    ##### SAVE TRAINED MODEL #####
    print("\nSaving trained model...")
    torch.save(model, "{}/trained_model.pkl".format(model_output_PATH))

    ##### SAVE TRAINING LOSS #####
    print("\nSaving training loss...")
    with open("{}/loss.txt".format(model_output_PATH), "w") as f:
        for i, loss in enumerate(training_loss):
            f.write("{},{}\n".format(i, loss))
            
    #### SAVE TRAINED EMBEDDINGS #####
    print("\nExtracing embeddings...")
    embeddings = {}
    model = torch.load("{}/trained_model.pkl".format(model_output_PATH))

    entity_embedding_tensor = model.entity_representations[0](
        indices=None).cpu().detach().numpy()
    relation_embedding_tensor = model.relation_representations[0](
    ).cpu().detach().numpy()

    embeddings["ent_embeddings"] = entity_embedding_tensor
    embeddings["rel_embeddings"] = relation_embedding_tensor

    print("Saving embeddings...")
    pickle.dump(embeddings,
                open("{}/{}.p".format(embedding_output_PATH,
                                      FLAGS.kge_triple_variants),
                     "wb"),
                protocol=4)

    print("{}/{}.p write success!\n".format(embedding_output_PATH,
          FLAGS.kge_triple_variants))
    print("All Done!")

if __name__ == "__main__":
    app.run(main)
