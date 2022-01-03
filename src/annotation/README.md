# Annotate

**annotate** is a C++ program that extracts almost all *decision-making points* from game records of Mahjong Soul, and converts the game situation at each decision-making point together with the player's choice and round's/game's final results into annotations suitable to learning. The output of the program is text, where each line represents the annotation of a decision-making point. The format of the annotation (each line of the output) is specified in the [Annotation Specification](../../#annotation-specification) section.

A *decision-making point* is defined by a point in a game at which a player is forced to choose an action among multiple options. Possible points and actions are enumerated as follows:

* Immediately after a self-draw (zimo, 自摸):
  * which tile in their hand to discard if not in riichi,
  * whether or not to declare riichi if possible,
  * whether or not to declare to win (zimo hu, 自摸和) if possible,
  * whether or not to declare a closed kong (an gang, 暗槓) if possible,
  * whether or not to declare a open kong (jia gang, 加槓) if possible, and
  * whether or not to declare no game (jiu zhong jiu pai, 九種九牌) if possible.
* Immediately after other player's discard:
  * whether or not to declare chi if possible,
  * whether or not to declare pon (peng, ポン, 碰) if possible,
  * whether or not to declare kong (da ming gang, 大明槓) if possible, and
  * whether or not to declare to win (rong, 栄和) if possible.
* Immediately after other player's kong:
  * whether or not declare to win (qiang gang, 槍槓) if possible.

However, decision-making points in the following relatively rare situations cannot be extracted and are assumed to be skipped, because they are not recorded in game records even if the player makes a choice, or the GUI disappears before the player makes a choice:

* A player is forced to choose whether or not to chow, but pon, kong, or rong by another player takes precedence, or
* a player is forced to choose whether or not to pon or kong, but rong by another player takes precedence.

## How to Build (For Developer Only)

Non-developers do not need to run the following command to build the Docker image because it built and available as a [public Docker image](https://hub.docker.com/r/cryolite/kanachan.annotate).

```bash
kanachan$ docker build -f annotate/Dockerfile -t cryolite/kanachan.annotate .
```

## How to Use

### Data Preparation

**annotate** assumes that game records of Mahjong Soul reside in one directory. All the files in that directory must be game record files. Each file in the directory must consists of the record of exactly one game. The content of each file, the format of the game record in other words, must be the same as the WebSocket response message that is returned from the Mahjong Soul API server when you hit a URL of the format `https://game.mahjongsoul.com/?paipu=YYMMDD-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`, where `YYMMDD-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX` is the UUID of a game.

### Annotation

Let `/path/to/data` be the path to the directory you have prepared according to the description in the [Data Preparation](#data-preparation) section. The following command line

```bash
$ docker run -i -v /path/to/data:/data:ro --rm cryolite/kanachan.annotate
```

prints the annotations converted from the game records in `/path/to/data` to the standard output.

Be careful not to specify the `-t` option to the `docker run` command, because the `-t` option will cause the newline code in the standard output of the `docker run` command to be `\r\n` instead of `\n` (cf. https://github.com/moby/moby/issues/8513). However, the training programs will work correctly even if the newline code in the annotation data is `\r\n`.
