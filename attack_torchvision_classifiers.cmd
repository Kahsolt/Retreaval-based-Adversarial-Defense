@ECHO OFF
REM cmdline examples

SET MODEL=resnet18

REM clean
python attack_torchvision_classifiers.py -M %MODEL%

REM attack only (default param setting)
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 10 --eps 8/255 --alpha 1/255
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 4  --eps 4/255 --alpha 1/255
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 1  --eps 1/255 --alpha 1/255
REM strict FGSM (no random start)
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 1  --eps 1/255 --alpha 1/255 -nrs

REM defense only (default param setting)
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 5.0
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 2.0
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 1.5
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 1.25
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 1.0
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 0.75
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 0.5
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 0.25

python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 3 --s 1.5
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 5 --s 1.5
python attack_torchvision_classifiers.py -M %MODEL% --dfn --k 7 --s 1.5

REM attack & defense
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 10 --eps 8/255 --alpha 1/255 --dfn --k 5 --s 1.5
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 4  --eps 4/255 --alpha 1/255 --dfn --k 5 --s 1.5
python attack_torchvision_classifiers.py -M %MODEL% --atk --step 10 --eps 8/255 --alpha 1/255 --dfn --k 3 --s 1.25
