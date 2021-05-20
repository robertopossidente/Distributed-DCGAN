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
5) Security Group: Foi criado um security group sem restrições, apenas para fins de estudo dessa atividade, sem compromisso com melhores práticas de segurança.
6) Placemente Group: Foram utilizados duas instâncias do tipo c5.xlarge para otimização da performance computacional da execução da aplicação DAG.
7) Foram utilizados os seguintes packages para utlização da aplicação mpirun: openmpi-bin libopenmpi-dev libhdf5-openmpi-dev 
8) Acesso às máquinas liberado através dos comandos: ssh-keygen e cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
9) Criação de alias permanente na instância: OBS: Configuração: --num_epochs 1, --nnodes=2, --nproc_per_node=4 e -master_addr="172.31.26.253" (IP obtido pelo AWS console)
   touch ~/.bash_aliases
   vi ~/.bash_aliases
   alias launch-gan='sudo alias launch-gan='sudo docker run --env OMP_NUM_THREADS=1 --rm --network=host -v=$(pwd):/root dist_dcgan:latest python -m torch.distributed.launch 
   --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="172.31.26.253" --master_port=1234 dist_dcgan.py --dataset cifar10 --dataroot ./cifar10 --num_epochs 1'
   source ~/.bashrc
10) Criação do arquivo hostfile com os seguintes comandos: (IPs obtido pelo AWS console)
   echo "<IP_Node1> slots=4" > hostfile
   echo "<IP_Node2> slots=4" > hostfile
11) Personalização do alias de execução da aplicação em cada uma das instâncas: 
   a) Instância Master -> --node_rank 0
   b) Instância Slave -> --node_rank 1
12) Execucão da DAG usando MPI -> mpirun -np 4 --hotfile hostfile launch-gan

Resultados: A aplicação DAG foi executado utilizando MPI através do procedimento descrito na seção de metodologia desse experimento baseados na adaptação dos passos do estudo de caso 2, descrito na Seção 3.3.4, do Capítulo 3 do livro intitulado "Minicursos do WSCAD 2018". Foi possível observar as duas máquinas, master com rank=0 e slave com rank=1, executando simultaneamente a aplicação e número total de iterações da época distribuídas entre cada uma, ou seja, de um total de 398 iterações, cada máquina ficou reponsável por 198 iterações. 

Análise: Através da aplicação mpirun foi possível verificar o efeito da execução paralela da aplicação DAG em dois nodes hosteados pela AWS. 
 
