import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.core.window import Window

Window.clearcolor = (0, 0.3, 0.6, 1)

class MyApp(App):
    def build(self):
        self.textinput = TextInput(text='', multiline=False)
        self.button = Button(text='Submit')
        self.button.bind(on_press=self.reaction)

        self.label = Label(text='')

        root_widget = Widget()
        root_widget.add_widget(self.textinput)
        root_widget.add_widget(self.button)
        root_widget.add_widget(self.label)
        return root_widget

    def reaction(self, instance):
        entered_text = self.textinput.text
        self.label.text = 'You said: {}'.format(entered_text)

if __name__=="__main__":
    MyApp().run()