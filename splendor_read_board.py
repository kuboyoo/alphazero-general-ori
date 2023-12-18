#!/usr/bin/env python3
import pandas as pd
import time
from datetime import datetime
from selenium import webdriver

from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
from selenium.common.exceptions import NoSuchElementException

from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

import yaml
import os
import multiprocessing

from splendor.SplendorPlayers import *
from splendor.SplendorGame import SplendorGame as Game
from splendor.NNet import NNetWrapper as NNet
from splendor.SplendorLogicNumba import Board
from utils import *
from MCTS import MCTS
from controlable_play import main as predict

#driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install())) #自動更新する場合

_lock = multiprocessing.Lock()
global game
game = None

def main():
    url = 'https://boardgamearena.com/10/splendor?table=426233203'
    target_id = 0 #推論対象のプレイヤー番号(0->先手, 1->後手)

    #ニューラルネットの初期化
    game = Game(2, False)
    model_name = "./Heian-kyo/genbu.pt"
    nn_args = dict(lr=None, dropout=0., epochs=None, batch_size=None, nn_version=-1, name=model_name)
    net = NNet(game, nn_args)
    cpt_dir, cpt_file = os.path.split(model_name)
    additional_keys = net.load_checkpoint(cpt_dir, cpt_file)
    cpuct = 2.5
    fpu = 0.3
    num_mcts = 16000
    mcts_args = dotdict({
    'numMCTSSims'     : num_mcts, # if args.numMCTSSims else additional_keys.get('numMCTSSims', 100),
    'cpuct'           : cpuct,
    'fpu'             : fpu,
    'prob_fullMCTS'   : 1.,
    'forced_playouts' : False,
    'no_mem_optim'    : False,
    })

    """
    game = Game(2, False) #for debug
    predict(game, mcts, net, target_id, None)
    MCTS.reset_all_search_trees()
    return
    """
    


    cards_dict =  {"card_1":"W311",
                    "card_2":"W22",
                    "card_3":"W3",
                    "card_4":"W21",
                    "card_5":"W221",
                    "card_6":"W2111",
                    "card_7":"W4",
                    "card_8":"W1111",
                    "card_9":"B21",
                    "card_10":"B2111",
                    "card_11":"B1111",
                    "card_12":"B221",
                    "card_13":"B311",
                    "card_14":"B4",
                    "card_15":"B22",
                    "card_16":"B3",
                    "card_17":"G4",
                    "card_18":"G22",
                    "card_19":"G3",
                    "card_20":"G311",
                    "card_21":"G2111",
                    "card_22":"G21",
                    "card_23":"G221",
                    "card_24":"G1111",
                    "card_25":"R221",
                    "card_26":"R311",
                    "card_27":"R21",
                    "card_28":"R22",
                    "card_29":"R2111",
                    "card_30":"R4",
                    "card_31":"R1111",
                    "card_32":"R3",
                    "card_33":"K4",
                    "card_34":"K221",
                    "card_35":"K311",
                    "card_36":"K3",
                    "card_37":"K2111",
                    "card_38":"K1111",
                    "card_39":"K22",
                    "card_40":"K21",
                    "card_41":"W322",
                    "card_42":"W332",
                    "card_43":"W421",
                    "card_44":"W5",
                    "card_45":"W53",
                    "card_46":"W6",
                    "card_47":"B332",
                    "card_48":"B322",
                    "card_49":"B53",
                    "card_50":"B421",
                    "card_51":"B5",
                    "card_52":"B6",
                    "card_53":"G6",
                    "card_54":"G5",
                    "card_55":"G53",
                    "card_56":"G421",
                    "card_57":"G332",
                    "card_58":"G322",
                    "card_59":"R332",
                    "card_60":"R322",
                    "card_61":"R421",
                    "card_62":"R53",
                    "card_63":"R5",
                    "card_64":"R6",
                    "card_65":"K322",
                    "card_66":"K332",
                    "card_67":"K421",
                    "card_68":"K5",
                    "card_69":"K53",
                    "card_70":"K6",
                    "card_71":"W7",
                    "card_72":"W633",
                    "card_73":"W5333",
                    "card_74":"W73",
                    "card_75":"B633",
                    "card_76":"B73",
                    "card_77":"B7",
                    "card_78":"B5333",
                    "card_79":"G7",
                    "card_80":"G633",
                    "card_81":"G5333",
                    "card_82":"G73",
                    "card_83":"R73",
                    "card_84":"R633",
                    "card_85":"R7",
                    "card_86":"R5333",
                    "card_87":"K7",
                    "card_88":"K633",
                    "card_89":"K73",
                    "card_90":"K5333",
                    }

    nobles_dict = {"noble_1":"RG",
                "noble_2":"BG",
                "noble_3":"BW",
                "noble_4":"KW",
                "noble_5":"KR",
                "noble_6":"KBW",
                "noble_7":"KRG",
                "noble_8":"KRW",
                "noble_9":"GBR",
                "noble_10":"GBW",
                }

    #chromeでアクセス
    options = webdriver.ChromeOptions()
    service = webdriver.chrome.service.Service(executable_path = '/opt/homebrew/bin/chromedriver')
    #driver = webdriver.Chrome(service=service, options=options)
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) #自動更新する場合
    driver.get(url)
    time.sleep(5)

    overall = driver.find_element(By.XPATH, '//*[@id="overall-content"]')

    players_part = overall.find_element(By.XPATH, '//*[@id="right-side-first-part"]')
    player_names = players_part.find_elements(By.XPATH, './/*[contains(@class, "player-name")]')
    player1_name = player_names[0].text
    player2_name = player_names[1].text

    player_1_reserve = []
    player_2_reserve = []

    while True:
        
        print("player1の予約枠",player_1_reserve)
        print("player2の予約枠",player_2_reserve)

        #ループ開始時に予約リストを初期化
        reserve_dict = {player1_name:[],player2_name:[]}
        
        #まず予約枠を埋め尽くす
        logs = driver.find_element(By.XPATH, '//*[@id="logs"]')
        reserve_logs = logs.find_elements(By.XPATH, './/*[contains(@class, "spl_notif-inner-tooltip")]')
        for reserve_log in reserve_logs:
            parent_element = reserve_log.find_element(By.XPATH, './..')
            player_log = parent_element.find_element(By.XPATH, './span[@class="playername"]')
            
            reserve_card = "card_"+str(reserve_log.get_attribute('data-id'))
            reserve_player = player_log.get_attribute('innerHTML')
            reserve_dict[reserve_player].append(cards_dict[reserve_card])
            
        player_1_reserve = reserve_dict[player1_name]
        player_2_reserve = reserve_dict[player2_name]
            
        user_input = input("Enterで情報取得、endで終了 ")
        
        """
        user_input = input("Enterで情報取得、rで予約入力、endで終了 ")
        
        if user_input=="r":
            player_1_add = input("player_1の予約入力(ex. B633)、無い場合又はデッキトップはEnter ")
            if player_1_add != "":
                if player_1_add in cards_dict.values():
                    player_1_reserve.append(player_1_add)
                else:
                    print("入力が不正です")
                
            player_2_add = input("player_2の予約入力(ex. B633)、無い場合又はデッキトップはEnter ")
            if player_2_add != "":
                if player_2_add in cards_dict.values():
                    player_2_reserve.append(player_2_add)
                else:
                    print("入力が不正です")
            
            continue
        """
                
        if user_input == "end":
            break
        
        noblesbar_element = overall.find_element(By.XPATH, '//*[@id="noblesbar"]')
        div_elements = noblesbar_element.find_elements(By.XPATH, './div[position() <= 3]')
        div_ids = [div.get_attribute('id') for div in div_elements]
        div_transformed = [nobles_dict[div_id] for div_id in div_ids]
        #print(div_ids)
        #print(div_transformed)
        data = {"Nobles":div_transformed}

        card_board_element = overall.find_element(By.XPATH, '//*[@id="cards"]')
        card_elements = card_board_element.find_elements(By.XPATH, './/*[contains(@class, "spl_card spl_coloreditem")]')
        div_ids = [card_element.get_attribute('id') for card_element in card_elements]
        div_transformed = [cards_dict[div_id] for div_id in div_ids]
        #print(div_ids)
        #print(div_transformed)
        Tier3_list = div_transformed[:4]
        Tier2_list = div_transformed[4:8]
        Tier1_list = div_transformed[8:]
        data["Tier3"] = Tier3_list
        data["Tier2"] = Tier2_list
        data["Tier1"] = Tier1_list

        coinsbar_element = overall.find_element(By.XPATH, '//*[@id="coinsbar"]')
        coincounter_elements = coinsbar_element.find_elements(By.XPATH, './/*[contains(@class, "coinpile_counter")]')

        text_list = [element.text for element in coincounter_elements]
        # BWKRGg 104325 を並べ替える必要がある
        order = [1,0,4,3,2,5]  # 並べ替える順序のインデックスリスト
        text_list = [int(text_list[i]) for i in order]
        #print(text_list)
        data["Bank"]=text_list

        player_board_element = overall.find_element(By.XPATH, '//*[@id="player_boards"]')
        player_board_elements = player_board_element.find_elements(By.XPATH, './/*[contains(@class, "player-board")]')
        player_board_element_1 = player_board_elements[0]
        player_board_element_2 = player_board_elements[1]

        player_card_elements = player_board_element_1.find_elements(By.XPATH, './/*[contains(@class, "spl_number")]')
        div_ids = [div.get_attribute('class') for div in player_card_elements]
        #print("div_ids: ", div_ids)
        player1_cardtools = []
        player1_coins = []
        for i, item in enumerate(div_ids):
            if item.endswith('depleted'):
                value = 0
            else:
                value = int(item.split('_')[-1])

            if i % 2 == 0:
                if i != 10:
                    player1_cardtools.append(value)
                else:
                    player1_coins.append(value)
            else:
                player1_coins.append(value)
        
        player_card_elements = player_board_element_2.find_elements(By.XPATH, './/*[contains(@class, "spl_number")]')
        div_ids = [div.get_attribute('class') for div in player_card_elements]
        #print("div_ids: ", div_ids)
        player2_cardtools = []
        player2_coins = []
        for i, item in enumerate(div_ids):
            if item.endswith('depleted'):
                value = 0
            else:
                value = int(item.split('_')[-1])

            if i % 2 == 0:
                if i != 10:
                    player2_cardtools.append(value)
                else:
                    player2_coins.append(value)
            else:
                player2_coins.append(value)
                
        data["Gems"]=[player1_coins,player2_coins]
        data["Cards"]=[player1_cardtools,player2_cardtools]
        
        player_noble_elements = player_board_element_1.find_elements(By.XPATH, './/*[contains(@class, "spl_noble")]')
        div_ids = [div.get_attribute('id') for div in player_noble_elements]
        #print(div_ids)
        player1_nobles = div_ids
        player1_nobles = [nobel.replace("mininoble", "noble") for nobel in player1_nobles]
        player1_nobles = [nobles_dict[nobel] for nobel in player1_nobles]

        player_noble_elements = player_board_element_2.find_elements(By.XPATH, './/*[contains(@class, "spl_noble")]')
        div_ids = [div.get_attribute('id') for div in player_noble_elements]
        #print(div_ids)
        player2_nobles = div_ids
        player2_nobles = [nobel.replace("mininoble", "noble") for nobel in player2_nobles]
        player2_nobles = [nobles_dict[nobel] for nobel in player2_nobles]
        
        data["PlayersNobles"]=[player1_nobles,player2_nobles]
        
        actions = ActionChains(driver)
        #所持カード取得
        original_elements = player_board_element_1.find_elements(By.XPATH, './/*[contains(@class,"spl_cardcount")]')
        print("Getting player1 card_contents...")
        player1_card_contents = []
        for original_element in original_elements:
            actions.move_to_element(original_element).perform()
            time.sleep(0.7)  # 待機
            
            try:
                target_element = driver.find_element(By.XPATH, '//*[@id="dijit__MasterTooltip_0"]/div[2]')
                card_tools = target_element.find_elements(By.XPATH, './/*[contains(@class,"spl_card spl_coloreditem")]')
                for card_tool in card_tools:
                    card_id = card_tool.get_attribute('id')
                    player1_card_contents.append(cards_dict[card_id])
                    #print(card_id)
                    if cards_dict[card_id] in player_1_reserve:
                        player_1_reserve.remove(cards_dict[card_id])
        
            #player1のカードが無い場合に発生する例外
            except NoSuchElementException:
                continue
        
        original_elements = player_board_element_2.find_elements(By.XPATH, './/*[contains(@class,"spl_cardcount")]')
        print("Getting player2 card_contents...")
        player2_card_contents = []
        for original_element in original_elements:
            actions.move_to_element(original_element).perform()
            time.sleep(0.7)  # 待機
            
            try:
                target_element = driver.find_element(By.XPATH, '//*[@id="dijit__MasterTooltip_0"]/div[2]')
                card_tools = target_element.find_elements(By.XPATH, './/*[contains(@class,"spl_card spl_coloreditem")]')
                for card_tool in card_tools:
                    card_id = card_tool.get_attribute('id')
                    player2_card_contents.append(cards_dict[card_id])
                    #print(card_id)
                    if cards_dict[card_id] in player_2_reserve:
                        player_2_reserve.remove(cards_dict[card_id])
        
            #player1のカードが無い場合に発生する例外
            except NoSuchElementException:
                continue
        
        data["PlayersCards"]=[player1_card_contents,player2_card_contents]
        data["Reserve"] = [player_1_reserve, player_2_reserve]
        
        
        # 現在の日時を取得
        current_time = datetime.now()

        # ファイル名を作成
        file_name = f"log/output_{current_time.strftime('%Y%m%d_%H%M%S')}.yaml"

        # YAML形式でデータを書き出す
        with open(file_name, 'w') as file:
            yaml.dump(data, file, sort_keys=False, default_flow_style=False)
        game = Game(2, False)
        mcts = MCTS(game, net, mcts_args)
        predict(game, mcts, net, target_id, data)
        MCTS.reset_all_search_trees()
        
    driver.quit()

if __name__ == "__main__":
    main()