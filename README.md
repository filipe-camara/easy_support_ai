Easy Support AI 🚀
Sistema inteligente e automatizado para suporte técnico de TI utilizando Inteligência Artificial de última geração, combinando Classificação de Urgência de Machine Learning tradicional com busca semântica em base de documentos usando RAG (Retrieval-Augmented Generation) e orquestração de fluxo de agentes com LangGraph.

1. Problema Resolvido
O atendimento de suporte técnico de TI tradicional sofre com alta latência de resposta, falta de padronização, gargalos no triagem manual e escalonamento indevido ou demorado para níveis de suporte sênior.

O Easy Support AI resolve esse problema ao:

Classificar Automaticamente a Urgência de novos tickets utilizando algoritmos de Machine Learning treinados localmente.
Responder de Forma Contextualizada dúvidas operacionais comuns baseando-se estritamente em uma base de conhecimento corporativa segura (RAG).
Decidir se deve Abster-se de responder caso o assunto não conste nos manuais oficiais ou o usuário necessite de escalonamento especializado (nível 2, nível 3 ou segurança).
2. Arquitetura
O ecossistema é composto pelas seguintes camadas técnicas integradas:

Camada de Dados: Arquivos markdown estruturados contendo as políticas de TI (knowledge_base/) e banco de dados SQLite (data/support.db) para registros estruturados.
Classificador de Urgência: Modelo de vetorização de texto TfidfVectorizer combinado com classificador Supervisionado (LinearSVC ou LogisticRegression), treinado localmente para separar tickets de urgência alta, média e baixa.
Módulo de RAG (Busca Vetorial): Utilização da API de Embeddings da OpenAI (text-embedding-3-small) acoplada ao banco vetorial Chroma para indexação e recuperação robusta por similaridade de cosseno com cálculo de relevância métrica.
Orquestrador de Fluxos (LangGraph): Grafo de decisão automatizado que gerencia os estados do ticket, chama o classificador, invoca o retriever RAG, analisa o orçamento de tokens e formata a resposta final ao cliente.
Interface Gráfica (Streamlit): Front-end interativo que permite ao usuário final simular a abertura de chamados e visualizar o diagnóstico inteligente em tempo real.
3. Pré-requisitos
Para executar o projeto no Linux ou Google Colab, você precisará de:

Python 3.10 ou superior
NPM / Node.js (necessário para expor o servidor local usando localtunnel)
Chave de API da OpenAI ativa
4. Instalação
Para instalar as dependências necessárias no Linux, execute os seguintes comandos no terminal:

# Clonar ou acessar a pasta do repositório
cd /content/easy_support_ai

# Atualizar o pip e instalar as dependências obrigatórias
pip install -U pip
pip install -r requirements.txt
5. Variáveis de Ambiente
Crie um arquivo .env na raiz do projeto contendo as seguintes configurações obtidas a partir de seu .env.example:

OPENAI_API_KEY=sua_chave_real_da_openai_aqui
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
RAG_TOP_K=5
RAG_MIN_SCORE=0.45
MAX_CONTEXT_TOKENS=3500
CLASSIFIER_CONFIDENCE_MIN=0.65
6. Preparação dos Dados
O projeto utiliza o arquivo data/tickets_treino.csv contendo 60 exemplos anotados divididos estrategicamente para garantir generalização do modelo:

Urgência Alta: Chamados impeditivos críticos de indisponibilidade de produção e segurança.
Urgência Média: Lentidão em telas, falhas de integração secundárias ou lentidão de relatórios.
Urgência Baixa: Dúvidas frequentes de configuração de conta, localização de guias ou dúvidas em geral.
7. Treinamento
Para treinar o classificador de urgência, execute:

cd /content/easy_support_ai
PYTHONPATH=. python -m classifier.train
O script avaliará as métricas contra um baseline aleatório e salvará o classificador pronto em classifier/urgency_model.joblib.

8. Indexação
Alimente o banco vetorial a partir dos manuais contidos na base de conhecimento executando o script de indexação:

cd /content/easy_support_ai
PYTHONPATH=. python -m rag.index_documents
9. Execução
Para rodar a interface Streamlit interativa e expor um túnel seguro para acessar no seu navegador local:

# 1. Iniciar o Streamlit em segundo plano
nohup streamlit run app.py --server.port 8501 &

# 2. Expor via localtunnel
npx localtunnel --port 8501
Digite o IP público exibido no comando curl ipv4.icanhazip.com como senha ao acessar o link gerado pelo localtunnel.

10. Testes
Para rodar a suíte completa de testes automatizados e validar o integrador RAG e classificação, execute:

cd /content/easy_support_ai
PYTHONPATH=. python -m pytest
11. Métricas
O modelo possui avaliação automática das métricas:

Acurácia e F1-Macro do Classificador: Atualmente com F1-Macro atingindo ~`0.61no conjunto de validação, superando largamente o baseline de0.16`.
Avaliação de RAG (Abstenção e Precisão): Testes de cobertura integrados avaliando se perguntas fora de escopo são corretamente ignoradas sem gerar alucinações.
12. Segurança
O sistema preserva a segurança e a integridade de dados:

Isolamento de Chaves: Chaves sensíveis ficam armazenadas em variáveis de ambiente .env e nunca expostas no código de acordo com o .gitignore.
Filtros de Redação: Garantia de tratamento apropriado sem vazamento de logs ou informações corporativas confidenciais para redes não autorizadas.
13. Limitações
O modelo de classificação atual é puramente linear baseado em TF-IDF, o que pode não capturar nuances profundas de contexto comparado a redes neurais complexas.
O banco vetorial local (Chroma) é executado em memória temporária para este ambiente Colab, necessitando de reindexação ao recriar a instância.
14. Melhorias Futuras
Implementar ajuste fino (fine-tuning) usando LLMs de classificação.
Integrar uma base de dados externa persistente como PostgreSQL com PGVector.
Adicionar monitoramento contínuo em tempo real via LangSmith para rastrear latências e custos de tokens por sessão.
