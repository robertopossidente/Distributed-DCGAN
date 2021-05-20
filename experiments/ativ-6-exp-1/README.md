Data: 20/05/2021 - Roberto de Oliveira Possidente - r225554@g.unicamp.br

Atividade 6: Semana 06/05 - Executando a Rede Generativa Adversarial

Objetivo
O objetivo desta atividade é aprender a utilizar a nuvem computacional para execução de uma aplicação de maneira distribuída.

Experimento: Adaptar os passos do estudo de caso 2, descrito na Seção 3.3.4, do Capítulo 3 do livro intitulado "Minicursos do WSCAD 2018" para treinar a Rede Generativa Adversarial, 
utilizada na atividade 5, na nuvem computacional. 

Metodogoia: Adaptação dos passos relatados no estudo de caso 2 da Seção 3.3.4:
1) AMI: Ubuntu 20.04 LTS 
2) Tipo de instância EC2: t2.2xlarge -> Escolhida para servir de "image template" para a AMI a ser gerada. Melhor custo benefíco, considerando requesitos de multiplos cores, 
4 no caso, e memória de pelo menos 16Gb. OBS: Requisitos definidos arbitrariamente baseados na atividade 5.
3) Security Group: Foi cricado um security group sem restrições, apenas para fins de estudo dessa atividade, sem compromisso com melhores práticas de segurança.
4) Foram utilizados os seguintes packages para utlização da aplicação mpirun: openmpi-bin libopenmpi-dev libhdf5-openmpi-dev 
5) Acesso às máquinas liberado través dos comandos: ssh-keygen e cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
6) Criação de alias:
7) Criação do arquivo hostfile: 
8) Excucão da DAG usando MPI -> mpirun -np 4 --hotfile hostfile launch-gan
 
