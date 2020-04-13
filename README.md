Wasserstein GAN
===============

論文 ["Wasserstein GAN"](https://arxiv.org/abs/1701.07875)に付随するコード

## いくつかの注意事項

- LSUNデータセットの初回実行時には、データローダの作成に長い時間（最大1時間）がかかることがあります。最初に実行した後は小さなキャッシュファイルが作成され、数秒で処理が完了します。キャッシュはlmdbデータベース（LSUN）のインデックスのリストです。
- コードへの唯一の追加は（忘れていましたが、論文へは後で追加します）[main.pyの163-166行目](https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py#L163-L166)です。まず前提としてDiscriminatorの学習回数の方がGeneratorの学習回数より多いです。割合はoption(D.iters)で設定できますが、デフォルトでは5回です。で話を戻すとこれらの行は、ジェネレータの学習の最初のイテレーションの25回目か、もしくはとても散発的に(500回のジェネレータのイテレーションにつき1回)動作します。やることは、Discriminatorの更新回数をこのときだけめっちゃ増やすと言うことです。Discriminatorの反復回数をデフォルトの5回ではなく100回に設定しています。これは、最初の反復でもDiscriminatorを最適な状態で開始するのに役立ちます。パフォーマンスに大きな違いはないはずですが、特に学習曲線を視覚化するときに役立ちます（そうしないと、Discriminatorが適切に訓練されるまで損失が大きくなるのがわかるからです）。これが、最初の25回のイテレーションが残りの訓練よりもかなり長い時間を要する理由でもあります。
- もしあなたの学習曲線が突然大きく落ちた場合は、[これ](https://github.com/martinarjovsky/WassersteinGAN/issues/2)を見てみてください。これは、criticが最適に近づくことができず、つまりcriticの学習がうまくいかずWasserstein距離の推定が上手くできなくなってしまったと考えられます。既知の原因は、高い学習率と大きなmomentum(慣性)であり、ceiticを軌道に戻すのを助けるものは何でも、この問題を解決する可能性があります。

## 前提条件

- Linux もしくは OSXのコンピュータであること
- [PyTorch](http://pytorch.org)
- 学習には、速度の面からNVIDIA GPUの使用を強く推奨します。CPUでもサポートされていますが、学習は非常に遅くなります。

二つの主な経験談:

### ジェネレータのサンプル品質(c.f.生成画像の質)はDiscriminatorの損失と相関します。

![gensample](imgs/w_combined.png "sample quality correlates with discriminator loss")

### モデルの安定性の向上

![stability](imgs/compare_dcgan.png "stability")


## LSUN実験の再現

**With DCGAN:**

```bash
python main.py --dataset cifar10 --dataroot /home/data/Tate/ --cuda
```

**With MLP:**

```bash
python main.py --mlp_G --ngf 512
```

生成されたサンプルは `samples` フォルダにあります。

Loss_D` の値をプロットすると、論文の曲線を再現することができます。論文の曲線には(論文で述べられているように)中央値フィルターが適用されています。

```python
med_filtered_loss = scipy.signal.medfilt(-Loss_D, dtype='float64'), 101)
```

より改良されたREADMEを公開しました。
