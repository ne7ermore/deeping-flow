import argparse
import time
import os
import random
from termcolor import colored

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from common import middle_load, middle_save, set_logger
from data_loader import DataLoader
from model import Transformer
from extract_feature import BertVector
from valid import test

parser = argparse.ArgumentParser(description='poiter network')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--pre_epochs', type=int, default=20)
parser.add_argument('--test_start_epochs', type=int, default=80)
parser.add_argument('--cuda_device', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--logdir', type=str,
                    default='/home/notebook/data/personal/taojian/logdir')
parser.add_argument('--model_path', type=str,
                    default='/home/notebook/data/personal/taojian/weights')
parser.add_argument('--tag', type=str, default='pt')
parser.add_argument('--data', type=str, default=f'corpus')
parser.add_argument('--dropout', type=float, default=0.9)
parser.add_argument('--d_model', type=int, default=512)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--n_head', type=int, default=8)
parser.add_argument('--d_k', type=int, default=64)
parser.add_argument('--d_v', type=int, default=64)
parser.add_argument('--enc_stack_layers', type=int, default=2)
parser.add_argument('--dec_stack_layers', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--pretrain_lr', type=float, default=0.0001)
parser.add_argument('--dist_model', type=int, default=768)
parser.add_argument('--dist_rate', type=float, default=0.4)
parser.add_argument('--dist_encode_rate', type=float, default=0.5)

parser.add_argument('--not_use_pretrain', action='store_true')
parser.add_argument('--use_debug', action='store_true')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.compat.v1.set_random_seed(args.seed)

args.turn_size = 6
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

data = middle_load(args.data)
args.vocab_size = len(data["word2idx"])
args.max_context_len = data["max_len"]

args.use_pretrain = not args.not_use_pretrain

training_data = DataLoader(
    data["train"]["src_texts"],
    data["train"]["src_turn"],
    data["train"]["tgt_indexs"],
    data["train"]["tgt_texts"],
    data["train"]["eos_indexs"],
    data["train"]["src_context"],
    data["train"]["tgt_context"],
    batch_size=args.batch_size)

validation_data = DataLoader(
    data["valid"]["src_texts"],
    data["valid"]["src_turn"],
    data["valid"]["tgt_indexs"],
    data["valid"]["tgt_texts"],
    data["valid"]["eos_indexs"],
    data["valid"]["src_context"],
    data["valid"]["tgt_context"],
    batch_size=args.batch_size)

model = Transformer(args)

saver = tf.train.Saver()
sv = tf.train.Supervisor(logdir=f"{args.logdir}_{args.tag}_{args.cuda_device}",
                         saver=saver,
                         global_step=model.global_step,
                         summary_op=None)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

best_score = -1

bert = BertVector(max_seq_len=args.max_context_len)

with sv.managed_session(config=config) as sess:
    if args.use_debug:
        model.debug(training_data, sess, bert)
    else:
        os.makedirs(
            f"{args.model_path}_{args.tag}_{args.cuda_device}", exist_ok=True)
        os.makedirs(
            f"{args.model_path}_best_{args.tag}_{args.cuda_device}", exist_ok=True)

        logger = set_logger(colored('poiter network', 'yellow'),
                            usefile=f"{args.model_path}_{args.tag}_{args.cuda_device}/log")
        logger.info("params info")
        for k, v in args.__dict__.items():
            logger.info(f"{k} - {v}")
        logger.info("-"*90)

        save_data = {
            "word2idx": data["word2idx"],
            "max_context_len": data["max_len"],
            "turn_size": args.turn_size,
        }
        middle_save(
            save_data, f"{args.model_path}_{args.tag}_{args.cuda_device}/corpus")
        os.system(
            f"cp {args.model_path}_{args.tag}_{args.cuda_device}/corpus {args.model_path}_best_{args.tag}_{args.cuda_device}/")

        if args.use_pretrain:
            logger.info("Pretrain...")
            for epoch in range(1, args.pre_epochs + 1):
                training_data_tqdm = tqdm(
                    training_data, mininterval=1, desc=f'pre-train Processing (loss=X.XXXX)', leave=False)
                logger.info(
                    f"pre-train epoch {epoch}/{args.pre_epochs} loss: {model.pre_train(training_data_tqdm, sess, bert, training_data.batch_size)/training_data.stop_step:.4f}")

            logger.info("Backup pretrain model")
            os.system(
                f"cp -rf {args.logdir}_{args.tag}_{args.cuda_device} {args.logdir}_{args.tag}_{args.cuda_device}_pretrain")

        logger.info("Train...")
        for epoch in range(1, args.epochs + 1):
            if sv.should_stop():
                break

            training_data_tqdm = tqdm(
                training_data, mininterval=1, desc='train Processing (loss=X.XXXX correct=XXX\XXX)', leave=False)
            total_loss, total_correct, total_gold, rouge_scores = model.train(
                training_data_tqdm, sess, bert, training_data.batch_size)
            logger.info(f"train epoch {epoch}/{args.epochs} loss: {total_loss/training_data.stop_step:.4f} correct: {total_correct} gold count: {total_gold} presicion: {total_correct/total_gold:.4f} rouge score: {rouge_scores/training_data.sents_size:.4f}")

            validation_data_tqdm = tqdm(
                validation_data, mininterval=1, desc='valid Processing (loss=X.XXXX correct=XXX\XXX)', leave=False)
            total_loss, total_correct, total_gold, rouge_scores = model.valid(
                validation_data_tqdm, sess, bert, validation_data.batch_size)
            os.system(
                f"rm {args.model_path}_{args.tag}_{args.cuda_device}/model.*")
            saver.save(
                sess, f"{args.model_path}_{args.tag}_{args.cuda_device}/model")
            logger.info(f"valid epoch {epoch}/{args.epochs} loss: {total_loss/validation_data.stop_step:.4f} correct: {total_correct} gold count: {total_gold} presicion: {total_correct/total_gold:.4f} rouge score: {rouge_scores/validation_data.sents_size:.4f}")

            if epoch > args.test_start_epochs:
                cc, pc = test(
                    f"{args.model_path}_{args.tag}_{args.cuda_device}", best_score, args.data)
                score = cc / pc
                logger.info(
                    f"test epoch {epoch}/{args.epochs} precision: {score:.4f} {cc}/{pc}")

                if score > best_score:
                    logger.info(
                        f"new best presicion score {score:.4f} and save model")
                    best_score = score
                    os.system(
                        f"cp {args.model_path}_{args.tag}_{args.cuda_device}/model* {args.model_path}_best_{args.tag}_{args.cuda_device}/")
