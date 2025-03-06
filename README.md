# genai-workspace

> 모든 과정은 S-Core 사내에서 사용하는 것을 전제로 했다.
> minikube를 기준으로 했지만, docker desktop kubernetes에서도 정상적으로 동작한다.

## minikube 설치

minikube 문서를 참고해서 minikube를 설치한다.

> https://minikube.sigs.k8s.io/docs/start/

기본적으로 플랫폼에 맞는 minikube 실행파일을 받은 후, 파일명을 minikube (.exe) 로 변경하여 path가 잡힌 디렉토리에 복사해 넣으면 된다.

## minikube 실행

minikube 설치 후, minikube에 인증서를 넣어 준다. (사전에 인증서를 미리 넣어두지 않으면 ingress가 정상적으로 동작하지 않는다.)

```sh
$ mkdir -p ~/.minikube/certs/
$ cp __custom__.crt ~/.minikube/certs/
```

minikube를 시작한다. 시스템 자원은 상황에 맞춰 넉넉하게 지정해 준다.

```sh
$ export HTTPS_PROXY=http://30.30.30.27:8080
$ minikube start --addons=ingress,dashboard --driver=docker --memory=8GB --cpus=4 --disk-size=30GB
```

## 개발툴 컨테이너 빌드

Code-server와 JupyterLab 등의 개발툴 컨테이너들을 빌드한다.

만약 공용 image repository에 올려서 사용할 것이라면 그냥 빌드해서 올려놓으면 되지만, 만약 로컬에서만 진행할 거라면 이미지 빌드를 minikube 상에서 진행하거나 docker 환경을 minikube로 만든 후 빌드해야 한다.

```sh
$ eval $(minikube docker-env)
```

Windows의 경우엔 eval 명령이 없기에 minikube docker-env로 출력 된 명령어를 직접 복사해 쓰면 된다.

* chroma-ui
```sh
$ cd deploy/container/chroma-ui
$ docker build -t chroma-ui . --progress=plain
```

* code-server
```sh
$ cd deploy/container/code-server_for_genai
$ docker build -t code-server_for_genai . --progress=plain
```

* jupyterlab
```sh
$ cd deploy/container/jupyter_for_genai
$ docker build -t jupyter_for_genai . --progress=plain
```

로컬에서만 진행할 거라면 deployment에서 pullPullPolicy를 IfNotPresent로 지정해 준다.

## K8S에 배포

helm 설치 후, helm 명령어로 배포 시작

```sh
$ cd deploy/helm
$ helm install genai-graph -n genai --create-namespace
```

## 서비스 노출

minikube 터널링을 동작시킨다. 이때 bind-address를 0.0.0.0으로 지정해 줘야 외부에서 접속이 가능해진다. (생략할 경우 디폴트는 127.0.0.1이라서 현재 로컬 PC에서만 접속이 가능하다.)

```sh
$ minikube tunnel --bind-address=0.0.0.0
```

## 개별 PC 설정

리눅스의 경우 /etc/hosts, 윈도우의 경우 %WinDir%/System32/drivers/etc/hosts 파일에 각 호스트들의 주소를 minikube가 동작하는 시스템에 맞춰 입력해 준다. (로컬의 경우엔 127.0.0.1로 지정한다.)

```
  ...

# for genai test
127.0.0.1  neo4j.local neo4j-browser.local chromadb.local chroma-ui.local flowise.local jupyter.local code-server.local

  ...
```

로컬의 경우, proxy에 해당 호스트에 대한 예외를 추가해 준다.

```
*.local
```

## 제공 서비스들

| 서비스 | 클러스터 내부 주소 | 외부 서비스 주소 |
|-------|-------------------|-----------------|
| Neo4j | http://svc-neo4j:7687 <br/> http://svc-neo4j-browser:7474 | http://neo4j.local:80 <br/> http://neo4j-browser.local |
| ChromaDB | http://svc-chromadb:8000 <br/> http://svc-chroma-ui:3000 | http://chromadb.local <br/> http://chroma-ui.local |
| Flowise | http://svc-flowise:3000 | http://flowise.local |
| JupyterLab | http://svc-jupyter:8888 | http://jupyter.local |
| Code Server | http://svc-code-server:8443 | http://code-server.local |
