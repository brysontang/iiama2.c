# TEV accuracy: models/stories15M.bin

dim 288, hidden 768, 6 layers, vocab 32000, int8 group size 32. Reference: 6 stories sampled from fp32 at temperature 1, 1154 positions, replayed through each backend.

## Classifier on the TEV, int8 hidden states

recall@K is how often the int8 CPU's best token lands in the TEV's top K.

| TEV classifier | recall@1 | @4 | @16 | @64 | top-p nucleus in top 64 |
|---|---|---|---|---|---|
| blend:16:0.25 | 74.0% | 95.4% | 99.8% | 100.0% | 95.1% |
| clamp8:16:0.5 | 91.9% | 99.9% | 100.0% | 100.0% | 97.8% |
| exact11:8:1.0 | 99.1% | 100.0% | 100.0% | 100.0% | 98.0% |

## End to end, teacher-forced against fp32

| backend | top-1 vs fp32 | KL (nats) | ppl | TEV saturation |
|---|---|---|---|---|
| fp32 CPU | 100.0% | 0.0000 | 2.77 | - |
| int8 CPU (the Wii today) | 98.4% | 0.0005 | 2.77 | - |
| split (blend/16 cols, headroom 0.25, CPU top-256) | 98.4% | 0.0046 | 2.78 | cls 7.19% |
| GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256 | 92.2% | 0.0273 | 2.84 | wq 0.30%, wk 0.29%, wv 0.25%, wo 0.26%, w1 0.31%, w2 0.02%, w3 0.30%, cls 6.34% |
| GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256 | 97.0% | 0.0054 | 2.78 | wq 0.00%, wk 0.00%, wv 0.00%, wo 0.00%, w1 0.00%, w2 0.00%, w3 0.00%, cls 7.06% |
| GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256 | 90.0% | 0.0398 | 2.88 | wq 1.82%, wk 1.82%, wv 1.60%, wo 1.72%, w1 1.82%, w2 0.48%, w3 1.80%, cls 6.57% |

## Greedy stories

### "Once upon a time"

**fp32 CPU**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to knock the ball down.
>
> Lily found a stick and tried to hit the ball. But the stick was

**int8 CPU (the Wii today)**

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a long stick and tried to reach the ball. She poked

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a long stick and tried to reach the ball. She poked

**GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256** (diverges from int8 CPU at token 34)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, dark cloud in the sky. She knew that meant rain was coming.
>
> Lily ran inside her house and told her mommy about the rain. Her mommy said, "Let's make some hot cocoa to warm up!" Lily loved hot cocoa, so she said, "Yay! I love hot cocoa!"
>
> They made hot cocoa together and sat down to drink it. Lily's

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 109)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a stick and tried to reach the ball. She poked the

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 34)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, dark cloud in the sky. She knew that meant rain was coming. 
>
> Lily ran inside her house and told her mommy about the rain. Her mommy said, "Let's make some hot cocoa to warm up!" Lily loved hot cocoa. 
>
> They made hot cocoa and drank it while they watched the rain. Lily felt warm and cozy. She was happy that she got

### "The grumpy turtle king"

**fp32 CPU**

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's father came to the forest and saw the king crying. He asked the

**int8 CPU (the Wii today)**

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's friend, a little rabbit, came to see what was wrong. The

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's friend, a little rabbit, came to see what was wrong. The

**GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256** (diverges from int8 CPU at token 17)

> The grumpy turtle king was walking in the forest. He was very tired and wanted to rest. He saw a big tree and decided to rest under it. The tree was very old and had many leaves. The king king sat down and closed his eyes.
>
> Suddenly, he heard a loud noise. He opened his eyes and saw a big bird flying towards him. The bird was very hungry and wanted to eat the king king. The king king was very scared and didn't know what to do. He tried to run away, but the bird was

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 11)

> The grumpy turtle king was walking along the beach. He was looking for something to do. Suddenly, he saw a little girl. She was playing in the sand and she was making a sand castle.
>
> The king said to the little girl, "What are you doing?"
>
> The little girl said, "I'm making a castle!"
>
> The king said, "That's very nice. Can I help you?"
>
> The little girl said, "Yes, please!"
>
> So the king and the little girl worked together to make the castle. They used

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 11)

> The grumpy turtle king was walking along the beach. He was looking for something to eat. Suddenly, he saw a big, juicy apple. He was so excited that he quickly grabbed it.
>
> The king said, "I'm going to eat this apple!" But the apple was too big and he couldn't eat it. He tried to pull it out of the sand, but it was too heavy.
>
> The king was so angry that he shouted, "I will eat this apple!" He started to eat the apple. But the apple
