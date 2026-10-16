# TEV accuracy: models/stories15M.bin

dim 288, hidden 768, 6 layers, vocab 32000, int8 group size 32. Reference: 6 stories sampled from fp32 at temperature 1, 1154 positions, replayed through each backend.

## Classifier on the TEV, int8 hidden states

recall@K is how often the int8 CPU's best token lands in the TEV's top K.

| TEV classifier | recall@1 | @4 | @16 | @64 | top-p nucleus in top 64 |
|---|---|---|---|---|---|
| blend:16:0.25 | 73.1% | 95.1% | 99.8% | 100.0% | 93.9% |
| clamp8:16:0.5 | 91.9% | 99.9% | 100.0% | 100.0% | 97.2% |
| exact11:8:1.0 | 99.1% | 100.0% | 100.0% | 100.0% | 97.4% |

## End to end, teacher-forced against fp32

| backend | top-1 vs fp32 | KL (nats) | ppl | TEV saturation |
|---|---|---|---|---|
| fp32 CPU | 100.0% | 0.0000 | 2.71 | - |
| int8 CPU (the Wii today) | 98.5% | 0.0007 | 2.72 | - |
| split (blend/16 cols, headroom 0.25, CPU top-256) | 98.5% | 0.0047 | 2.72 | cls 7.08% |
| GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256 | 92.7% | 0.0293 | 2.81 | wq 0.30%, wk 0.29%, wv 0.25%, wo 0.27%, w1 0.31%, w2 0.02%, w3 0.30%, cls 6.50% |
| GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256 | 97.6% | 0.0057 | 2.73 | wq 0.00%, wk 0.00%, wv 0.00%, wo 0.00%, w1 0.00%, w2 0.00%, w3 0.00%, cls 6.96% |
| GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256 | 91.6% | 0.0393 | 2.82 | wq 1.82%, wk 1.82%, wv 1.60%, wo 1.72%, w1 1.82%, w2 0.48%, w3 1.80%, cls 6.64% |

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
> Lily found a stick and tried to reach the ball. She poked the

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily wanted to play with the ball, but it was too high up in the sky. She tried to jump and reach it, but she couldn't. Then, she had an idea. She would use a stick to get the ball down.
>
> Lily found a stick and tried to reach the ball. She poked the

**GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256** (diverges from int8 CPU at token 35)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red apple on the ground. She picked it up and took a bite. It was so juicy and delicious!
>
> Lily's mom came outside and saw her eating the apple. "What are you doing, Lily?" she asked. "I found a big, red apple," Lily replied. "Can I have a bite?" her mom asked. "Sure!" Lily said.

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 55)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty.
>
> Lily's mommy came outside and saw her looking up at the sky. "What are you looking at, Lily?" she asked. "The sun is rising!" Lily said. "It's so pretty!"
>
> Lily's mommy smiled and said, "Yes, it is. But it's

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 34)

> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, dark cloud in the sky. She knew that meant rain was coming. 
>
> Lily ran inside her house and told her mommy about the rain. Her mommy said, "Let's make some hot chocolate to warm up!" They went to the kitchen and made hot chocolate. It was so yummy! 
>
> After they finished their hot chocolate, Lily's mommy said,

### "The grumpy turtle king"

**fp32 CPU**

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's father came to the forest and saw the king crying. He asked the

**int8 CPU (the Wii today)**

> The grumpy turtle king was walking in the forest. He was very tired and wanted to rest. He saw a big tree and decided to rest under it. The tree was very tall and had many leaves. The king sat down and closed his eyes.
>
> Suddenly, he heard a loud noise. It was a big bird who wanted to eat the king. The king woke up and saw the bird. He was very scared and tried to run away, but the bird was too fast. The bird caught the king and ate him. The end.

**split (blend/16 cols, headroom 0.25, CPU top-256)** (same as int8 CPU)

> The grumpy turtle king was walking in the forest. He was very tired and wanted to rest. He saw a big tree and decided to rest under it. The tree was very tall and had many leaves. The king sat down and closed his eyes.
>
> Suddenly, he heard a loud noise. It was a big bird who wanted to eat the king. The king woke up and saw the bird. He was very scared and tried to run away, but the bird was too fast. The bird caught the king and ate him. The end.

**GPU only (clamp8/16 cols, headroom 0.5, row scales), blended classifier + CPU top-256** (same as int8 CPU)

> The grumpy turtle king was walking in the forest. He was very tired and wanted to rest. He saw a big tree and decided to rest under it. The tree was very tall and had many leaves. The king sat down and closed his eyes.
>
> Suddenly, he heard a loud noise. It was a big bird who wanted to eat the king. The king woke up and saw the bird. He was very scared and tried to run away, but the bird was too fast. The bird caught the king and ate him. The end.

**GPU only (exact11/8 cols, headroom 1.0), blended classifier + CPU top-256** (diverges from int8 CPU at token 17)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of food. Suddenly, he heard a loud noise. He looked up and saw a big bird flying above him. The bird was carrying a big bag of food.
>
> The king was very angry. He shouted at the bird and said, "That's my food! Give it back!" The bird just flew away. The king was so angry that he started to cry.
>
> The king's father came to the forest and saw the king crying. He asked the

**GPU only (clamp8/16 cols, headroom 0.5), blended classifier + CPU top-256** (diverges from int8 CPU at token 17)

> The grumpy turtle king was walking in the forest. He was carrying a big bag of treasure. Suddenly, he heard a loud noise. He looked up and saw a big bird flying in the sky. The bird was carrying something in its beak.
>
> The king was very angry. He shouted at the bird and said, "What are you doing with my treasure?" The bird replied, "I'm going to give it to my friend. He will be very happy."
>
> The king was very angry. He shouted, "No! You
