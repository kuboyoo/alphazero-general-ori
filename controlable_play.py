#!/usr/bin/env python3

from review import review
import Arena
from MCTS import MCTS
#from santorini.SantoriniPlayers import *
#from santorini.SantoriniGame import SantoriniGame as Game
#from santorini.NNet import NNetWrapper as NNet

from splendor.SplendorPlayers import *
from splendor.SplendorGame import SplendorGame as Game
from splendor.NNet import NNetWrapper as NNet
from splendor.SplendorLogic import np_all_nobles, np_all_cards_1, np_all_cards_2, np_all_cards_3, len_all_cards, np_different_gems_up_to_2, np_different_gems_up_to_3, np_2specs_gems_up_to_3, np_cards_symmetries, np_reserve_symmetries
from splendor.SplendorLogicNumba import Board

import numpy as np
from utils import *
import os.path
from os import stat
import subprocess
import itertools
import json
import multiprocessing
import copy
import yaml

game = None
_lock = multiprocessing.Lock()

"""
対話的な符号入力による盤面制御（オンラインプレイ用）
"""

def yml2board(game, yml, curPlayer):
  color_map = {
    "B": 0,
    "R": 1,
    "K": 2,
    "W": 3,
    "G": 4
  }
  cost_map = {
    ("Tier1", "W"): {
        "3":   0,
        "21":  1,
        "22":  2,
        "221": 3,
        "311": 4,
        "1111":5,
        "2111":6,
        "4":   7
    },

    ("Tier2", "W"): {
        "322":0,
        "332":1,
        "5":  2,
        "53": 3,
        "421":4,
        "6":  5
    },

    ("Tier3", "W"): {
        "5333":0,
        "633":2,
        "7":  1,
        "73": 3
    },

    ("Tier1", "B"): {
        "3":   0,
        "21":  1,
        "22":  2,
        "221": 3,
        "311": 4,
        "1111":5,
        "2111":6,
        "4":   7
    },

    ("Tier2", "B"): {
        "322":0,
        "332":1,
        "5":  2,
        "53": 3,
        "421":4,
        "6":  5
    },

    ("Tier3", "B"): {
        "5333":0,
        "633":2,
        "7":  1,
        "73": 3
    },

    ("Tier1", "G"): {
        "3":   0,
        "21":  1,
        "22":  2,
        "221": 3,
        "311": 4,
        "1111":5,
        "2111":6,
        "4":   7
    },

    ("Tier2", "G"): {
        "322":0,
        "332":1,
        "5":  2,
        "53": 3,
        "421":4,
        "6":  5
    },

    ("Tier3", "G"): {
        "5333":0,
        "633":2,
        "7":  1,
        "73": 3
    },

    ("Tier1", "R"): {
        "3":   0,
        "21":  1,
        "22":  2,
        "221": 3,
        "311": 4,
        "1111":5,
        "2111":6,
        "4":   7
    },

    ("Tier2", "R"): {
        "322":0,
        "332":1,
        "5":  2,
        "53": 3,
        "421":4,
        "6":  5
    },

    ("Tier3", "R"): {
        "5333":0,
        "633":2,
        "7":  1,
        "73": 3
    },

    ("Tier1", "K"): {
        "3":   0,
        "21":  1,
        "22":  2,
        "221": 3,
        "311": 4,
        "1111":5,
        "2111":6,
        "4":   7
    },
    
    ("Tier2", "K"): {
        "322":0,
        "332":1,
        "5":  2,
        "53": 3,
        "421":4,
        "6":  5
    },

    ("Tier3", "K"): {
        "5333":0,
        "633":2,
        "7":  1,
        "73": 3
    }
  }

  cost_map_for_rsv = {
    "W3":   ("Tier1", 0),
    "W21":  ("Tier1", 1),
    "W22":  ("Tier1", 2),
    "W221": ("Tier1", 3),
    "W311": ("Tier1", 4),
    "W1111":("Tier1", 5),
    "W2111":("Tier1", 6),
    "W4":   ("Tier1", 7),

    "W322":("Tier2", 0),
    "W332":("Tier2", 1),
    "W5":  ("Tier2", 2),
    "W53": ("Tier2", 3),
    "W421":("Tier2", 4),
    "W6":  ("Tier2", 5),

    "W5333":("Tier3",0),
    "W633":("Tier3",2),
    "W7":  ("Tier3",1),
    "W73": ("Tier3",3),

    "B3":   ("Tier1",0),
    "B21":  ("Tier1",1),
    "B22":  ("Tier1",2),
    "B221": ("Tier1",3),
    "B311": ("Tier1",4),
    "B1111":("Tier1",5),
    "B2111":("Tier1",6),
    "B4":   ("Tier1",7),

    "B322":("Tier2",0),
    "B332":("Tier2",1),
    "B5":  ("Tier2",2),
    "B53": ("Tier2",3),
    "B421":("Tier2",4),
    "B6":  ("Tier2",5),
    
    "B5333":("Tier3", 0),
    "B633":("Tier3", 2),
    "B7":  ("Tier3", 1),
    "B73": ("Tier3", 3),

    "G3":   ("Tier1", 0),
    "G21":  ("Tier1", 1),
    "G22":  ("Tier1", 2),
    "G221": ("Tier1", 3),
    "G311": ("Tier1", 4),
    "G1111":("Tier1", 5),
    "G2111":("Tier1", 6),
    "G4":   ("Tier1", 7),

    "G322":("Tier2", 0),
    "G332":("Tier2", 1),
    "G5":  ("Tier2", 2),
    "G53": ("Tier2", 3),
    "G421":("Tier2", 4),
    "G6":  ("Tier2", 5),

    "G5333":("Tier3", 0),
    "G633": ("Tier3", 2),
    "G7":  ("Tier3", 1),
    "G73": ("Tier3", 3),

    "R3":   ("Tier1", 0),
    "R21":  ("Tier1", 1),
    "R22":  ("Tier1", 2),
    "R221": ("Tier1", 3),
    "R311": ("Tier1", 4),
    "R1111": ("Tier1", 5),
    "R2111": ("Tier1", 6),
    "R4":   ("Tier1", 7),

    "R322":("Tier2", 0),
    "R332":("Tier2", 1),
    "R5":  ("Tier2", 2),
    "R53": ("Tier2", 3),
    "R421":("Tier2", 4),
    "R6":  ("Tier2", 5),

    "R5333": ("Tier3", 0),
    "R633": ("Tier3", 2),
    "R7":  ("Tier3", 1),
    "R73": ("Tier3", 3),

    
    "K3":   ("Tier1", 0),
    "K21":  ("Tier1", 1),
    "K22":  ("Tier1", 2),
    "K221": ("Tier1", 3),
    "K311": ("Tier1", 4),
    "K1111":("Tier1", 5),
    "K2111":("Tier1", 6),
    "K4":   ("Tier1", 7),
    
    "K322": ("Tier2", 0),
    "K332": ("Tier2", 1),
    "K5":   ("Tier2", 2),
    "K53": ("Tier2", 3),
    "K421": ("Tier2", 4),
    "K6":  ("Tier2", 5),

    "K5333":("Tier3", 0),
    "K633": ("Tier3", 2),
    "K7":  ("Tier3", 1),
    "K73": ("Tier3", 3)
  }

  noble_map = {
      "RG": 0,
      "KR": 1,
      "BG": 2,
      "KW": 3,
      "BW": 4,
      "KRW":5,
      "GBW":6,
      "KRG":7,
      "GBR":8,
      "KBW":9
  }
  
  num_players = 2
  

  cardss = [
     [(color_map[cid[0]], cost_map[("Tier1", cid[0])][cid[1:]]) for cid in yml["Tier1"]],
     [(color_map[cid[0]], cost_map[("Tier2", cid[0])][cid[1:]]) for cid in yml["Tier2"]],
     [(color_map[cid[0]], cost_map[("Tier3", cid[0])][cid[1:]]) for cid in yml["Tier3"]]
  ]

  #場にカードを埋めていく処理
  for tier, cards in enumerate(cardss):
    for col_index, (color, card_index) in enumerate(cards):
      card  = game.board._get_select_card(tier, color, card_index)
      game.board.cards_tiers[8*tier+2*col_index:8*tier+2*col_index+2] = card

  game.board.bank[:] = yml["Bank"] + [0]
  for i in range(num_players):
    game.board.players_gems[i]  = np.array(yml["Gems"][i] + [0], np.int8)
    game.board.players_cards[i] = np.array(yml["Cards"][i] + [0, 0], np.int8)
    
    rsv_cards = yml["Reserve"][i]
    if len(rsv_cards) > 0:
      for j, cid in enumerate(rsv_cards):
        deck_name, id = cost_map_for_rsv[cid]
        color_id = color_map[cid[0]]
        if deck_name == "Tier3":
          rsv_card = game.board._get_select_card(2, color_id, id)
        elif deck_name == "Tier2":
          rsv_card = game.board._get_select_card(1, color_id, id)
        else:
          rsv_card = game.board._get_select_card(0, color_id, id)
        game.board.players_reserved[6*i+2*j:6*i+2*j+2, :] = rsv_card
    
    buyed_cards = yml["PlayersCards"][i]
    if len(buyed_cards) > 0:
      for j, cid in enumerate(buyed_cards):
        deck_name, id = cost_map_for_rsv[cid]
        color_id = color_map[cid[0]]
        if deck_name == "Tier3":
          card = game.board._get_select_card(2, color_id, id)
        elif deck_name == "Tier2":
          card = game.board._get_select_card(1, color_id, id)
        else:
          card = game.board._get_select_card(0, color_id, id)
        
        game.board.players_cards[i, 6] += card[1, 6]
  
    players_nobles = yml["PlayersNobles"][i] #プレイヤー取得済み貴族の登録
    if len(players_nobles) > 0:
      for j, noble in enumerate(players_nobles):
        #貴族判定時に添字被りを回避するため末尾から追加
        game.board.players_nobles[game.board.num_nobles*i+num_players-j] = np_all_nobles[noble_map[noble]]

  #for i_noble in range(game.board.num_nobles):
    #game.board.nobles[i_noble] = 0
  for i, n in enumerate(yml["Nobles"]):
    if n is not None:
      game.board.nobles[i, :] = np_all_nobles[noble_map[n]]

  print_board(game.board)
  #print("state: ", game.board.get_state())
  
  return game.getCanonicalForm(game.board.get_state(), curPlayer)

def main(game, mcts, net, curPlayer, board_yml=None): 
  if board_yml is None:
    with open('log/output_20231008_120039-b.yaml') as file:
    #with open("log/output_20230926_213919.yaml") as file:
      board_yml = yaml.safe_load(file)
    print(board_yml)

  board = yml2board(game, board_yml, curPlayer)

  model_name = "./Heian-kyo/genbu.pt"

  print(f"Player {curPlayer} 's. turn ...")

  
  review(board, game, curPlayer, mcts, net, temp=1, mode=0)

if __name__ == "__main__":
  game = Game(2, False)
  model_name = "./Heian-kyo/genbu.pt"
  nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1, name=model_name)
  net = NNet(game, nn_args)
  cpt_dir, cpt_file = os.path.split(model_name)
  additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
  cpuct = 2.5
  fpu = 0.3
  num_mcts = 10000
  mcts_args = dotdict({
    'numMCTSSims'     : num_mcts, # if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
    'cpuct'           : cpuct       if cpuct       else additional_keys.get('cpuct'      , 1.0),
    'fpu'             : fpu,
    'prob_fullMCTS'   : 1.,
    'forced_playouts' : False,
    'no_mem_optim'    : False,
  })
  mcts = MCTS(game, net, mcts_args)
  curPlayer = 1
  main(game, mcts, net, curPlayer)