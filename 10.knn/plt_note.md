






在使用`plt.imshow()`函数显示图像时，可以通过`cmap`参数来设置颜色映射（colormap）。`cmap`参数用于指定图像颜色的映射方式，不同的颜色映射可以用于突出显示不同的特征或属性。

`plt.cm.summer`是Matplotlib库中的一个内置颜色映射对象，它代表了一种夏季相关的颜色映射。夏季颜色映射通常用于表示温度或高度等与温暖、阳光相关的特性。

例如，可以使用以下代码将某个二维数组（例如图像）使用夏季颜色映射显示出来：

```python
import matplotlib.pyplot as plt

# 创建一个随机的二维数组
image_array = np.random.rand(100, 100)

# 使用plt.imshow()显示二维数组，并使用夏季颜色映射
plt.imshow(image_array, cmap=plt.cm.summer)
plt.show()
```

这将使用夏季颜色映射将图像的较低值映射为较暗的颜色，较高值映射为较亮的颜色。你可以根据需要选择不同的颜色映射，Matplotlib提供了许多其他的内置颜色映射，如'hot'、'cool'、'jet'等。