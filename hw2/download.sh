wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TElYcYNWtl8Uml0nVfbEKpZlVqIbcEMJ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TElYcYNWtl8Uml0nVfbEKpZlVqIbcEMJ" -O models_tokenizers_and_data.zip && rm -rf /tmp/cookies.txt

wait

unzip models_tokenizers_and_data.zip