#!/bin/bash
wget https://www.3c-tools.com/markets/bot-assist-explorer?list=binance_spot_usdt_galaxy_score_rank -O /tmp/binance_spot_usdt_galaxy_score_rank &>/dev/null
grep USDT_ /tmp/binance_spot_usdt_galaxy_score_rank |awk -F\> '{print $3}'|awk -F\< '{print $1}'|head -100|jq -R -s -c 'split("\n")[:-1]'

